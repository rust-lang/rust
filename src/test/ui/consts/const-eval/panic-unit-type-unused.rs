// compile-flags: --emit=mir,link

// Variant of panic-never-type.rs.
// Ensure that mir opts don't hide errors due to the usage of erroneous constants,
// in unused code, even for inhabited ZST constants.

#![warn(const_err)]
#![feature(const_panic)]

const UNIT: () = panic!();
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn foo() {
    let _ = UNIT;
    //~^ ERROR erroneous constant used
}

fn main() {}
