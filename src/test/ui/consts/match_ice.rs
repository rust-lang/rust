// compile-pass
// https://github.com/rust-lang/rust/issues/53708

struct S;

fn main() {
    const C: &S = &S;
    match C {
        C => {}
    }
}
