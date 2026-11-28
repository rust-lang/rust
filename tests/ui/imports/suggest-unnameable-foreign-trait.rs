//@ aux-crate:unnameable_traits=unnameable-traits.rs

fn main() {
    let _ = unnameable_traits::leak();

    0u8.unnameable_method();
    //~^ ERROR no method named `unnameable_method` found
    0u8.leaked_method();
    //~^ ERROR no method named `leaked_method` found
    0u8.reexported_method();
    //~^ ERROR no method named `reexported_method` found
}
