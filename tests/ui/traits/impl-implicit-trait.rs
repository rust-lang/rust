//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


enum option_<T> {
    none_,
    some_(T),
}

impl<T> option_<T> {
    pub fn foo(&self) -> bool { true }
}

enum option__ {
    none__,
    some__(isize)
}

impl option__ {
    pub fn foo(&self) -> bool { true }
}

pub fn main() {
}
