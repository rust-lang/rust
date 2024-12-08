#![warn(clippy::panic_in_result_fn)]
#![allow(clippy::unnecessary_wraps)]
struct A;

impl A {
    fn result_with_panic() -> Result<bool, String> // should emit lint
    //~^ ERROR: used `panic!()` or assertion in a function that returns `Result`
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

    fn result_with_todo() -> Result<bool, String> // should emit lint
    {
        todo!("Finish this");
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

    fn other_with_todo() // should not emit lint
    {
        todo!("finish this")
    }

    fn result_without_banned_functions() -> Result<bool, String> // should not emit lint
    {
        Ok(true)
    }
}

fn function_result_with_panic() -> Result<bool, String> // should emit lint
//~^ ERROR: used `panic!()` or assertion in a function that returns `Result`
{
    panic!("error");
}

fn in_closure() -> Result<bool, String> {
    let c = || panic!();
    c()
}

fn todo() {
    println!("something");
}

fn function_result_with_custom_todo() -> Result<bool, String> // should not emit lint
{
    todo();
    Ok(true)
}

fn issue_13381<const N: usize>() -> Result<(), String> {
    const {
        if N == 0 {
            panic!();
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    todo!("finish main method");
    Ok(())
}
