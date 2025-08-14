#![allow(dead_code)]
#![warn(clippy::redundant_test_prefix)]

fn main() {
    // Normal function, no redundant prefix.
}

fn f1() {
    // Normal function, no redundant prefix.
}

fn test_f2() {
    // Has prefix, but no `#[test]` attribute, ignore.
}

#[test]
fn test_f3() {
    //~^ redundant_test_prefix

    // Has prefix, has `#[test]` attribute. Not within a `#[cfg(test)]`.
    // No collision with other functions, should emit warning.
}

#[cfg(test)]
#[test]
fn test_f4() {
    //~^ redundant_test_prefix

    // Has prefix, has `#[test]` attribute, within a `#[cfg(test)]`.
    // No collision with other functions, should emit warning.
}

mod m1 {
    pub fn f5() {}
}

#[cfg(test)]
#[test]
fn test_f6() {
    //~^ redundant_test_prefix

    use m1::f5;

    f5();
    // Has prefix, has `#[test]` attribute, within a `#[cfg(test)]`.
    // No collision, has function call, but it will not result in recursion.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_foo_with_call() {
        //~^ redundant_test_prefix

        main();
    }

    #[test]
    fn test_f1() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f2() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f3() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f4() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f5() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f6() {
        //~^ redundant_test_prefix
    }
}

mod tests_no_annotations {
    use super::*;

    #[test]
    fn test_foo() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_foo_with_call() {
        //~^ redundant_test_prefix

        main();
    }

    #[test]
    fn test_f1() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f2() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f3() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f4() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f5() {
        //~^ redundant_test_prefix
    }

    #[test]
    fn test_f6() {
        //~^ redundant_test_prefix
    }
}

// This test is inspired by real test in `clippy_utils/src/sugg.rs`.
// The `is_in_test_function()` checks whether any identifier within a given node's parents is
// marked with `#[test]` attribute. Thus flagging false positives when nested functions are
// prefixed with `test_`. Therefore `is_test_function()` has been defined in `clippy_utils`,
// allowing to select only functions that are immediately marked with `#[test]` annotation.
//
// This test case ensures that for such nested functions no error is emitted.
#[test]
fn not_op() {
    fn test_not(foo: bool) {
        assert!(foo);
    }

    // Use helper function
    test_not(true);
    test_not(false);
}
