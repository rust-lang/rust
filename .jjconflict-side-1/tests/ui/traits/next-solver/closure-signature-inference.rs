//@ compile-flags: -Znext-solver
//@ check-pass

struct A;
impl A {
    fn hi(self) {}
}

fn hello() -> Result<(A,), ()> {
    Err(())
}

fn main() {
    let x = hello().map(|(x,)| x.hi());
}
