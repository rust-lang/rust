// When build the suggesttion take in consideration the `:?`
// https://github.com/rust-lang/rust/issues/100648
#![deny(warnings)]

fn main () {
    println!("hello {:?}", world = "world");
    //~^ ERROR named argument `world` is not used by name
    //~| HELP use the named argument by name to avoid ambiguity
    //~| SUGGESTION world
}
