// check-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

union X { a: i8 }

fn main() {
    let x = X { a: 5 };
    match x {
        X { a: _ | _ } => {},
    }
}
