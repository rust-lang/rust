//@ check-pass

trait Gen<T> {
    fn gen(x: Self) -> T;
}

struct A;

impl Gen<[(); 0]> for A {
    fn gen(x: Self) -> [(); 0] {
        []
    }
}

fn array() -> impl Gen<[(); 0]> {
    A
}

fn main() {
    let [] = Gen::gen(array());
}
