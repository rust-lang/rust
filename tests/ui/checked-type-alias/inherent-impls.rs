//@ check-pass

#![feature(checked_type_aliases)]
#![allow(incomplete_features)]

type Alias = Local;
struct Local;

impl Alias {
    fn method(self) {}
}

fn main() {
    let _ = Local.method();
    let _ = Local::method;
    let _ = Alias {}.method();
    let _ = Alias::method;
}
