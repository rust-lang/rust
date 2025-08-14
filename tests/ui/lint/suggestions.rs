// ignore-tidy-tab
//@ run-rustfix

#![warn(unused_mut, unused_parens)] // UI tests pass `-A unused`—see Issue #43896

#[no_mangle] const DISCOVERY: usize = 1;
//~^ ERROR const items should never be `#[no_mangle]`
//~| HELP try a static value

#[no_mangle]
//~^ HELP remove this attribute
pub fn defiant<T>(_t: T) {}
//~^ WARN functions generic over types or consts must be mangled

#[no_mangle]
fn rio_grande() {}

mod badlands {
    // The private-no-mangle lints shouldn't suggest inserting `pub` when the
    // item is already `pub` (but triggered the lint because, e.g., it's in a
    // private module). (Issue #47383)
    #[no_mangle] pub const DAUNTLESS: bool = true;
    //~^ ERROR const items should never be `#[no_mangle]`
    //~| HELP try a static value
    #[allow(dead_code)] // for rustfix
    #[no_mangle] pub fn val_jean<T>() {}
    //~^ WARN functions generic over types or consts must be mangled
    //~| HELP remove this attribute

    // ... but we can suggest just-`pub` instead of restricted
    #[no_mangle] pub(crate) const VETAR: bool = true;
    //~^ ERROR const items should never be `#[no_mangle]`
    //~| HELP try a static value
    #[allow(dead_code)] // for rustfix
    #[no_mangle] pub(crate) fn crossfield<T>() {}
    //~^ WARN functions generic over types or consts must be mangled
    //~| HELP remove this attribute
}

struct Equinox {
    warp_factor: f32,
}

fn main() {
    while true {
    //~^ WARN denote infinite loops
    //~| HELP use `loop`
        let mut registry_no = (format!("NX-{}", 74205));
        //~^ WARN does not need to be mutable
        //~| HELP remove this `mut`
        //~| WARN unnecessary parentheses
        //~| HELP remove these parentheses
        // the line after `mut` has a `\t` at the beginning, this is on purpose
        let mut
	        b = 1;
        //~^^ WARN does not need to be mutable
        //~| HELP remove this `mut`
        let d = Equinox { warp_factor: 9.975 };
        match d {
            #[allow(unused_variables)] // for rustfix
            Equinox { warp_factor: warp_factor } => {}
            //~^ WARN this pattern is redundant
            //~| HELP use shorthand field pattern
        }
        println!("{} {}", registry_no, b);
    }
}
