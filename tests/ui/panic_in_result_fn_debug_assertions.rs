#![warn(clippy::panic_in_result_fn)]
#![allow(clippy::unnecessary_wraps)]

struct A;

impl A {
    fn result_with_debug_assert_with_message(x: i32) -> Result<bool, String> // should emit lint
    {
        debug_assert!(x == 5, "wrong argument");
        Ok(true)
    }

    fn result_with_debug_assert_eq(x: i32) -> Result<bool, String> // should emit lint
    {
        debug_assert_eq!(x, 5);
        Ok(true)
    }

    fn result_with_debug_assert_ne(x: i32) -> Result<bool, String> // should emit lint
    {
        debug_assert_ne!(x, 1);
        Ok(true)
    }

    fn other_with_debug_assert_with_message(x: i32) // should not emit lint
    {
        debug_assert!(x == 5, "wrong argument");
    }

    fn other_with_debug_assert_eq(x: i32) // should not emit lint
    {
        debug_assert_eq!(x, 5);
    }

    fn other_with_debug_assert_ne(x: i32) // should not emit lint
    {
        debug_assert_ne!(x, 1);
    }

    fn result_without_banned_functions() -> Result<bool, String> // should not emit lint
    {
        let debug_assert = "debug_assert!";
        println!("No {}", debug_assert);
        Ok(true)
    }
}

fn main() {}
