#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

struct X { x: () }

fn main() {
    let x = Some(X { x: () });
    match x {
        Some(ref _y @ _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        None => panic!()
    }
}
