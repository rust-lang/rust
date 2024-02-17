//@ check-pass
union X { a: i8 }

fn main() {
    let x = X { a: 5 };
    match x {
        X { a: _ | _ } => {},
    }
}
