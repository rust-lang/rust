//@ check-pass
//@ compile-flags: -Znext-solver

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
