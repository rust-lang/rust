// https://github.com/rust-lang/rust/issues/53708

struct S;

#[derive(PartialEq, Eq)]
struct T;

fn main() {
    const C: &S = &S;
    match C {
        C => {}
        //~^ ERROR to use a constant of type `S` in a pattern, `S` must be annotated with
    }
    const K: &T = &T;
    match K { //~ ERROR non-exhaustive patterns: `&T` not covered
        K => {}
    }
}
