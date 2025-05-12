// #81282: Attributes are not allowed on struct field rest patterns (the ..).

struct S {}

fn main() {
    let S { #[cfg(any())] .. } = S {};
    //~^ ERROR expected identifier, found `..`
}
