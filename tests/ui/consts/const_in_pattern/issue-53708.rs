//@ check-pass
// https://github.com/rust-lang/rust/issues/53708
#[derive(PartialEq, Eq)]
struct S;

fn main() {
    const C: &S = &S;
    match C {
        C => {}
    }
}
