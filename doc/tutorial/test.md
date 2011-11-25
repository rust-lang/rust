# Testing

The Rust language has a facility for testing built into the language.
Tests can be interspersed with other code, and annotated with the
`#[test]` attribute.

    use std;
    
    fn twice(x: int) -> int { x + x }
    
    #[test]
    fn test_twice() {
        let i = -100;
        while i < 100 {
            assert twice(i) == 2 * i;
            i += 1;
        }
    }

When you compile the program normally, the `test_twice` function will
not be used. To actually run the tests, compile with the `--test`
flag:

    ## notrust
    > rustc --test twice.rs
    > ./twice
    running 1 tests
    test test_twice ... ok
    result: ok. 1 passed; 0 failed; 0 ignored

Or, if we change the file to fail, for example by replacing `x + x`
with `x + 1`:

    ## notrust
    running 1 tests
    test test_twice ... FAILED
    failures:
        test_twice
    result: FAILED. 0 passed; 1 failed; 0 ignored

You can pass a command-line argument to a program compiled with
`--test` to run only the tests whose name matches the given string. If
we had, for example, test functions `test_twice`, `test_once_1`, and
`test_once_2`, running our program with `./twice test_once` would run
the latter two, and running it with `./twice test_once_2` would run
only the last.

To indicate that a test is supposed to fail instead of pass, you can
give it a `#[should_fail]` attribute.

    use std;
    
    fn divide(a: float, b: float) -> float {
        if b == 0f { fail; }
        a / b
    }
    
    #[test]
    #[should_fail]
    fn divide_by_zero() { divide(1f, 0f); }

To disable a test completely, add an `#[ignore]` attribute. Running a
test runner (the program compiled with `--test`) with an `--ignored`
command-line flag will cause it to also run the tests labelled as
ignored.

A program compiled as a test runner will have the configuration flag
`test` defined, so that you can add code that won't be included in a
normal compile with the `#[cfg(test)]` attribute (see [conditional
compilation](syntax.md#conditional)).
