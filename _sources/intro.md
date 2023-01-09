<hr>

```{epigraph}
The two greatest inventions of the human mind are writing and money---the common language of intelligence, and the common language of self-interest.

--Victor de Riqueti, [marquis de Mirabeau](https://en.wikipedia.org/wiki/Victor_de_Riqueti,_marquis_de_Mirabeau)
```

```{epigraph}
He was one of those deep ones who know about stocks and shares. Of course no one really knows, but he quite seemed to know, and he often said stocks were up and shares were down in a way that would have made any woman respect him.

--J. M. Barrie, [*Peter Pan*](https://www.gutenberg.org/files/16/16-h/16-h.htm)
```

<hr>


# Introduction

This book project is currently under development. We will be using it in F-455 Quant Finance in Spring, 2023.

## Coding conventions

Even if a particular chapter doesn't have `import` statements, it is safe to assume that the following modules have been imported before executing any code.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import scipy.stats as scs

from numpy.random import default_rng
rng = default_rng(8675309)
```

## Greek letters in code

We often use Greek letters to denote variables or parameters in economics and finance. Throughout this book I'll use these letters in code, like this:

```python
μ = 0
σ = 1
```

I find this cleaner and easier to follow than the older coding convention, which was to write

```python
mu = 0
sigma = 1
```

To use Greek letters in your own code, you can copy them from the [Character viewer](https://support.apple.com/guide/mac-help/use-emoji-and-symbols-on-mac-mchlp1560/mac) on a Mac, or [Insert symbol](https://support.microsoft.com/en-us/topic/insert-ascii-or-unicode-latin-based-symbols-and-characters-d13f58d3-7bcb-44a7-a4d5-972ee12e50e0) on Windows. A Google search will turn up many options for making this process easier for whichever operating system you're using. Just be careful to be consistent when entering these letters, as there are different versions of the same thing: β and 𝛽 are two different symbols for beta, and using them both in code would create two different variables.

The Greek letters we use are:

Character | Name &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Character | Name
|---|---|---|---|
α|alpha    |μ|mu      
β|beta     |π|pi      
γ, Γ|gamma |ρ|rho     
δ, Δ|delta |σ, Σ|sigma
ε|epsilon  |τ|tau     
η|eta      |φ|phi     
θ|theta    |ω, Ω|omega
λ|lambda   |ν|nu      


# How to read this book

Items of particular importance will appear in one of the boxes shown here:

:::{note}
A clarifying note of special interest will appear in a box like this.
:::

:::{admonition} Key fact
This is an example of an important fact that you should pay careful attention to understand.
:::

:::{warning}
If it's easy to misuderstand a detail that may lead to confusion later, I'll note it in a warning box like this.
:::

:::{admonition} Extra credit
Details that require a little more math, or may be of interest only to especially motivated students, will appear in a box like this. And no, you don't actually receive extra credit for reading these.
:::

:::{admonition} Exercise
Exercises appear throughout the book.
:::

:::{admonition} Solution
:class: admonition-solution, dropdown
When a solution is provided, they will appear below the exercise and will be initially hidden. You should always attempt to complete the exercise before looking at the solution.
:::
