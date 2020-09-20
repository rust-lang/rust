// https://github.com/rust-lang/rust/issues/53708

struct S;

#[derive(PartialEq, Eq)]
struct T;

fn main() {
    const C: &S = &S;
    match C {
        //~^ non-exhaustive patterns: `&S` not covered
        C => {}
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN was previously accepted by the compiler
    }
    const K: &T = &T;
    match K {
        K => {}
    }
}
