#![warn(clippy::panic_in_result_fn)]
#![allow(clippy::uninlined_format_args, clippy::unnecessary_wraps)]

struct A;

impl A {
    fn result_with_assert_with_message(x: i32) -> Result<bool, String> // should emit lint
    //~^ panic_in_result_fn
    {
        assert!(x == 5, "wrong argument");
        Ok(true)
    }

    fn result_with_assert_eq(x: i32) -> Result<bool, String> // should emit lint
    //~^ panic_in_result_fn
    {
        assert_eq!(x, 5);
        Ok(true)
    }

    fn result_with_assert_ne(x: i32) -> Result<bool, String> // should emit lint
    //~^ panic_in_result_fn
    {
        assert_ne!(x, 1);
        Ok(true)
    }

    fn other_with_assert_with_message(x: i32) // should not emit lint
    {
        assert!(x == 5, "wrong argument");
    }

    fn other_with_assert_eq(x: i32) // should not emit lint
    {
        assert_eq!(x, 5);
    }

    fn other_with_assert_ne(x: i32) // should not emit lint
    {
        assert_ne!(x, 1);
    }

    fn result_without_banned_functions() -> Result<bool, String> // should not emit lint
    {
        let assert = "assert!";
        println!("No {}", assert);
        Ok(true)
    }
}

fn main() {}
