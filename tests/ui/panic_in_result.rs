#![warn(clippy::panic_in_result)]

struct A;

impl A {
    fn result_with_panic() -> Result<bool, String> // should emit lint
    {
        panic!("error");
    }

    fn result_with_unimplemented() -> Result<bool, String> // should emit lint
    {
        unimplemented!();
    }

    fn result_with_unreachable() -> Result<bool, String> // should emit lint
    {
        unreachable!();
    }

    fn option_with_unreachable() -> Option<bool> // should emit lint
    {
        unreachable!();
    }

    fn option_with_unimplemented() -> Option<bool> // should emit lint
    {
        unimplemented!();
    }

    fn option_with_panic() -> Option<bool> // should emit lint
    {
        panic!("error");
    }

    fn other_with_panic() // should not emit lint
    {
        panic!("");
    }

    fn other_with_unreachable() // should not emit lint
    {
        unreachable!();
    }

    fn other_with_unimplemented() // should not emit lint
    {
        unimplemented!();
    }

    fn result_without_banned_functions() -> Result<bool, String> // should not emit lint
    {
        Ok(true)
    }

    fn option_without_banned_functions() -> Option<bool> // should not emit lint
    {
        Some(true)
    }
}

fn main() {}
