//@ compile-flags: -Znext-solver
//@ check-pass

struct Struct {
    field: i32,
}

fn hello(f: impl Fn() -> &'static Box<[i32]>, f2: impl Fn() -> &'static Struct) {
    let cl = || {
        let x = &f()[0];
        let y = &f2().field;
    };
}

fn main() {}
