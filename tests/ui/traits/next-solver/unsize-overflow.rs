//@ compile-flags: -Znext-solver
#![recursion_limit = "8"]

fn main() {
    let _: Box<dyn Send> = Box::new(&&&&&&&1);
    //~^ ERROR overflow evaluating the requirement `Box<&&&&&&&i32>: CoerceUnsized<Box<dyn Send>>
}
