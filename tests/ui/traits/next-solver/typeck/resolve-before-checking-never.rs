//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(todo_macro_calls)]

#![feature(never_type)]

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

fn diverge() -> <! as Mirror>::Assoc { todo!() }

fn main() {
    let close = || {
        diverge();
    };
    let x: u32 = close();
}
