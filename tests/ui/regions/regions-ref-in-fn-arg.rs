#![feature(box_patterns)]


fn arg_item(box ref x: Box<isize>) -> &'static isize {
    x //~ ERROR cannot return value referencing function parameter
}

fn with<R, F>(f: F) -> R where F: FnOnce(Box<isize>) -> R { f(Box::new(3)) }

fn arg_closure() -> &'static isize {
    with(|box ref x| x) //~ ERROR cannot return value referencing function parameter
}

fn main() {}
