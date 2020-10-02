A `#![feature]` attribute was declared for a feature that is stable in the
current edition, but not in all editions.

Erroneous code example:

```ignore (limited to a warning during 2018 edition development)
#![feature(rust_2018_preview)]
#![feature(test_2018_feature)] // error: the feature
                               // `test_2018_feature` is
                               // included in the Rust 2018 edition
```
