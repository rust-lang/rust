#![feature(box_patterns)]
#![feature(box_syntax)]

fn arg_item(box ref x: Box<isize>) -> &'static isize {
    x //~^ ERROR borrowed value does not live long enough
}

fn with<R, F>(f: F) -> R where F: FnOnce(Box<isize>) -> R { f(box 3) }

fn arg_closure() -> &'static isize {
    with(|box ref x| x) //~ ERROR borrowed value does not live long enough
}

fn main() {}
