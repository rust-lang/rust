# 1st Report - ESOF

We are 4 students of the Engineering Faculty of the University of Oporto, and we're studying this project for the course of Software Engineering.

## Index

1. [RUST - Programming Language](#rust---programming-language)
2. [Project's Activity](#projects-activity)
3. [Developing Paradigm](#developing-paradigm)
    1. [Paradigm problems](#paradigm-problems)
    2. TAD (Test after develpment) vs TDD (Test driven development)

## RUST - Programming Language

Rust is a fast systems programming language that guarantees memory safety and offers painless concurrency (no data races). It does not employ a garbage collector and has minimal runtime overhead.

***

## Project's Activity
##### _Since 13/06/2010 Until 28/09/2015_

This project rellies on 1.164 contributors with a total of 46.757 commits.
In the last month it had 358 active pull requests. 322 of those are Merged pull request, and the other 36 are Proposed.
Regarding Issues there were 162 new ones and 259 closed.

***

## Developing Paradigm

Rust programming language uses **Test Driven Development - TDD** as its developing paradigm.

    The rhythm of Test-Driven Development (TDD) can be summed up as follows:
    1. Quickly add a test.
    2. Run all tests and see the new one fail.
    3. Make a little change.
    4. Run all tests and see them all succeed.
    5. Refactor to remove duplication.
###### Source: Kent Beck, “Test-Driven Development: By Example”, Addison-Wesley, 2002
***
### Paradigm problems

In contrast with _Rails_, _Rust_ has unit testing built right into the language (in which it is developed). However it’s just the basics: asserts and such. It's expected to have others added by the community as it grows.

### **TAD** _(Test after develpment)_ vs **TDD** _(Test driven development)_

|Test After Development|Test Driven Development|
|:---------------------|:----------------------|
|Allows some refactoring|Enables continual refactoring|
|Coverage levels up to ~75%|Coverage approaching 100%|
|No direct design impact|Driven the design|
|Can reduce defects|Significantly reduced defects, debugging cycles|
|Can be treated as separate task|Part of the coding process|
|-|Clarifies, documents understanding of requirements|
|-|Continual progress, consistent pacing|
|-|Continual feedback and learning|
|-|Sustainable|

###### Source: _Langr, Jeff. "Test-Driven Development vs. Test-After Development." N.p., n.d. Web. 28 Sept. 2015_
***
Acording to this study **Test Driven Develompent** is the best way to develop a project (at least comparing to TAD).
We agree with this judgement. In our experience we take longer to develop something without errors using TAD and write much more code, even knowing that TAD looks quicker.
