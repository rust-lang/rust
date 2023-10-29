#![crate_type = "lib"]

pub fn foo(callback: fn() -> dyn ToString) {
    //~^ ERROR: cannot have an unboxed trait object
    let mut x: Option<Box<dyn Fn() -> dyn ToString>> = None;
    x = Some(Box::new(callback));
}
