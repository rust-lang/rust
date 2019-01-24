// https://github.com/rust-lang/rust/issues/53708

struct S;

fn main() {
    const C: &S = &S;
    match C { //~ ERROR non-exhaustive
        C => {} // this is a common bug around constants and references in patterns
    }
}
