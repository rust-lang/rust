// run-pass
struct X { a: i32, b: i32 }

#[allow(unused_variables)]
fn main() {
    const DX: X = X { a: 0, b: 0 };
    const X1: X = X { a: 1, ..DX };  // ok
    let   x2    = X { a: 1, b: 2, ..DX };  // ok
    const X3: X = X { a: 1, b: 2, ..DX };  // error[E0016]
}
