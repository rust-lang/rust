// #81282: Attributes are not allowed on struct field rest patterns (the ..).

struct S {}

fn main() {
    let S { #[cfg(false)] .. } = S {};
    //~^ ERROR expected identifier, found `..`
}
