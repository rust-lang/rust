#![warn(clippy::panic_in_result_fn)]
#![allow(clippy::uninlined_format_args, clippy::unnecessary_wraps)]

// debug_assert should never trigger the `panic_in_result_fn` lint

struct A;

impl A {
    fn result_with_debug_assert_with_message(x: i32) -> Result<bool, String> {
        debug_assert!(x == 5, "wrong argument");
        Ok(true)
    }

    fn result_with_debug_assert_eq(x: i32) -> Result<bool, String> {
        debug_assert_eq!(x, 5);
        Ok(true)
    }

    fn result_with_debug_assert_ne(x: i32) -> Result<bool, String> {
        debug_assert_ne!(x, 1);
        Ok(true)
    }

    fn other_with_debug_assert_with_message(x: i32) {
        debug_assert!(x == 5, "wrong argument");
    }

    fn other_with_debug_assert_eq(x: i32) {
        debug_assert_eq!(x, 5);
    }

    fn other_with_debug_assert_ne(x: i32) {
        debug_assert_ne!(x, 1);
    }

    fn result_without_banned_functions() -> Result<bool, String> {
        let debug_assert = "debug_assert!";
        println!("No {}", debug_assert);
        Ok(true)
    }
}

fn main() {}
