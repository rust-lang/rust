//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

fn equal<T>(a: &T, b: &T) -> bool
where
    T: Eq,
{
    a == b
}

struct Struct;

fn main() {
    drop(equal(&Struct, &Struct))
    //~^ ERROR the trait bound `Struct: Eq` is not satisfied
}
