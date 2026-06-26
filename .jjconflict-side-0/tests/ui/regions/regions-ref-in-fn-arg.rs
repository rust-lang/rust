fn arg_item(ref x: Box<isize>) -> &'static isize {
    x //~ ERROR cannot return value referencing function parameter
}

fn with<R, F>(f: F) -> R where F: FnOnce(Box<isize>) -> R { f(Box::new(3)) }

fn arg_closure() -> &'static isize {
    with(|ref x| x) //~ ERROR cannot return value referencing function parameter
}

fn main() {}
