#![allow(dead_code)]

trait Deref {
    fn get(self) -> isize;
}

impl<'a> Deref for &'a isize {
    fn get(self) -> isize {
        *self
    }
}

fn with<R:Deref, F>(f: F) -> isize where F: FnOnce(&isize) -> R {
    f(&3).get()
}

fn return_it() -> isize {
    with(|o| o) //~ ERROR cannot infer
}

fn main() {
}
