% Unit testing

Unit tests should live in a `test` submodule at the bottom of the module they
test. Mark the `test` submodule with `#[cfg(test)]` so it is only compiled when
testing.

The `test` module should contain:

* Imports needed only for testing.
* Functions marked with `#[test]` striving for full coverage of the parent module's
  definitions.
* Auxiliary functions needed for writing the tests.

For example:

``` rust
// Excerpt from std::str

#[cfg(test)]
mod test {
    #[test]
    fn test_eq() {
        assert!((eq(&"".to_owned(), &"".to_owned())));
        assert!((eq(&"foo".to_owned(), &"foo".to_owned())));
        assert!((!eq(&"foo".to_owned(), &"bar".to_owned())));
    }
}
```

> **[FIXME]** add details about useful macros for testing, e.g. `assert!`
