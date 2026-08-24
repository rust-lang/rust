// Regression test for https://github.com/rust-lang/rust/issues/80954
// Rustc diagnostics should recognize that
// the assignment to 'x' and 'n' are being used
//@ check-pass

fn main() {
    let mut x = 0;
    match () {
        () if {
            x = 42;
            false
        } => {}
        _ => {
            println!("{}", x);
        }
    }

    let mut n = 0;
    match 0 {
        | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ if
        { println!("{}", n); n += 1; false } => {}
        _ => {}
    }
}
