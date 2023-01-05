#![crate_type = "lib"]

pub fn foo(callback: fn() -> dyn ToString) {
    let mut x: Option<Box<dyn Fn() -> dyn ToString>> = None;
    x = Some(Box::new(callback));
    //~^ ERROR: the size for values of type `dyn ToString` cannot be known at compilation time
}
