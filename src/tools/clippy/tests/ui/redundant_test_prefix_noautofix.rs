//@no-rustfix: name conflicts

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

fn f5() {}

#[cfg(test)]
#[test]
fn test_f5() {
    //~^ redundant_test_prefix

    // Has prefix, has `#[test]` attribute, within a `#[cfg(test)]`.
    // Collision with existing function.
}

mod m1 {
    pub fn f6() {}
    pub fn f7() {}
}

#[cfg(test)]
#[test]
fn test_f6() {
    //~^ redundant_test_prefix

    use m1::f6;

    f6();
    // Has prefix, has `#[test]` attribute, within a `#[cfg(test)]`.
    // No collision, but has a function call that will result in recursion.
}

#[cfg(test)]
#[test]
fn test_f8() {
    //~^ redundant_test_prefix

    use m1::f7;

    f7();
    // Has prefix, has `#[test]` attribute, within a `#[cfg(test)]`.
    // No collision, has function call, but it will not result in recursion.
}

// Although there's no direct call of `f` in the test, name collision still exists,
// since all `m3` functions are imported and then `map` is used to call `f`.
mod m2 {
    mod m3 {
        pub fn f(_: i32) -> i32 {
            0
        }
    }

    use m3::*;

    #[cfg(test)]
    #[test]
    fn test_f() {
        //~^ redundant_test_prefix
        let a = Some(3);
        let _ = a.map(f);
    }
}

mod m3 {
    fn test_m3_1() {
        // Has prefix, but no `#[test]` attribute, ignore.
    }

    #[test]
    fn test_m3_2() {
        //~^ redundant_test_prefix

        // Has prefix, has `#[test]` attribute. Not within a `#[cfg(test)]`.
        // No collision with other functions, should emit warning.
    }
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

    #[test]
    fn test_1() {
        //~^ redundant_test_prefix

        // `1` is invalid function name, so suggestion to rename is emitted
    }

    #[test]
    fn test_const() {
        //~^ redundant_test_prefix

        // `const` is reserved keyword, so suggestion to rename is emitted
    }

    #[test]
    fn test_async() {
        //~^ redundant_test_prefix

        // `async` is reserved keyword, so suggestion to rename is emitted
    }

    #[test]
    fn test_yield() {
        //~^ redundant_test_prefix

        // `yield` is reserved keyword for future use, so suggestion to rename is emitted
    }

    #[test]
    fn test_() {
        //~^ redundant_test_prefix

        // `` is invalid function name, so suggestion to rename is emitted
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

    #[test]
    fn test_1() {
        //~^ redundant_test_prefix

        // `1` is invalid function name, so suggestion to rename is emitted
    }

    #[test]
    fn test_const() {
        //~^ redundant_test_prefix

        // `const` is reserved keyword, so suggestion to rename is emitted
    }

    #[test]
    fn test_async() {
        //~^ redundant_test_prefix

        // `async` is reserved keyword, so suggestion to rename is emitted
    }

    #[test]
    fn test_yield() {
        //~^ redundant_test_prefix

        // `yield` is reserved keyword for future use, so suggestion to rename is emitted
    }

    #[test]
    fn test_() {
        //~^ redundant_test_prefix

        // `` is invalid function name, so suggestion to rename is emitted
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
