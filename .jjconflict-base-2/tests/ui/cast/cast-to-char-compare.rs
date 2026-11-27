//! regression test for <https://github.com/rust-lang/rust/issues/41998>
//@ check-pass

fn main() {
    if ('x' as char) < ('y' as char) {
        print!("x");
    } else {
        print!("y");
    }
}
