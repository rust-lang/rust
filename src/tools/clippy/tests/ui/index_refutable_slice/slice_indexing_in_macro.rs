#![deny(clippy::index_refutable_slice)]

extern crate if_chain;
use if_chain::if_chain;

macro_rules! if_let_slice_macro {
    () => {
        // This would normally be linted
        let slice: Option<&[u32]> = Some(&[1, 2, 3]);
        if let Some(slice) = slice {
            println!("{}", slice[0]);
        }
    };
}

fn main() {
    // Don't lint this
    if_let_slice_macro!();

    // Do lint this
    if_chain! {
        let slice: Option<&[u32]> = Some(&[1, 2, 3]);
        if let Some(slice) = slice;
        then {
            println!("{}", slice[0]);
        }
    }
}
