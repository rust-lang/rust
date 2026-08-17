//@ compile-flags: -Znext-solver -Awarnings
#![recursion_limit = "6"]

fn main() {
    let _: Box<dyn Send> = Box::new(&&&&&&&&&&&&1);
    //~^ ERROR overflow evaluating the requirement `Box<&&&&&&&&&&&&i32>: CoerceUnsized<Box<dyn Send>>
}
