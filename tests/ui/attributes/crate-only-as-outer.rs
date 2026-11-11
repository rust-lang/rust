#![deny(unused)]
#[crate_name = "owo"]
//~^ ERROR: crate-level attribute should be an inner attribute: add an exclamation mark: `#![crate_name]`

fn main() {}

mod inner {
    #![crate_name = "iwi"]
    //~^ ERROR: the `#![crate_name]` attribute can only be used at the crate root
}
