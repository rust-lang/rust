// https://github.com/rust-lang/rust/issues/53708

struct S;

#[derive(PartialEq, Eq)]
struct T;

fn main() {
    const C: &S = &S;
    match C {
        C => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN must be annotated
        //~| WARN previously accepted
    }
    const K: &T = &T;
    match K {
        K => {}
    }
}
