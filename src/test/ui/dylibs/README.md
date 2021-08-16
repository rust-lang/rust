This directory is a collection of tests of how dylibs and rlibs are compiled and loaded.

It is mostly an exploration checkpointing the current state of Rust.

We tend to advise people to try make at least one format available all the way
 up: e.g. all rlibs or all dylibs. But it is good to keep track of how we are
 handling other combinations.

There are seven auxiliary crates: `a`,`i`,`j`,`m`,`s`,`t`,`z`. Each top-level
test in this directory varies which of the auxiliary crates are compiled to
dylibs and which are compiled to rlibs.

The seven auxiliary form a dependence graph that looks like this (a pair of
diamonds):

```graphviz
   z -> s; s -> m; m -> i; i -> a
   z -> t; t -> m; m -> j; j -> a
// ~    ~~~~    ~~~~    ~~~~    ~
// |     |        |      |      |
// |     |        |      |      +- basement
// |     |        |      |
// |     |        |      +- ground
// |     |        |
// |     |        +- middle
// |     |
// |     +- upper
// |
// +- roof
```
