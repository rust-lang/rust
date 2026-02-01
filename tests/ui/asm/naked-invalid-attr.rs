//@ edition:2015
// Checks that the #[unsafe(naked)] attribute can be placed on function definitions only.
//
//@ needs-asm-support
//@ revisions: edition2015 edition2018
//@ [edition2015] edition: 2015
//@ [edition2018] edition: 2018..
#![unsafe(naked)] //~ ERROR attribute cannot be used on

use std::arch::naked_asm;

extern "C" {
    #[unsafe(naked)] //~ ERROR attribute cannot be used on
    fn f();
}

#[unsafe(naked)] //~ ERROR attribute cannot be used on
#[repr(C)]
struct S {
    #[unsafe(naked)] //~ ERROR attribute cannot be used on
    a: u32,
    b: u32,
}

trait Invoke {
    #[unsafe(naked)] //~ ERROR attribute cannot be used on
    extern "C" fn invoke(&self);
}

impl Invoke for S {
    #[unsafe(naked)]
    extern "C" fn invoke(&self) {
        naked_asm!("")
    }
}

#[unsafe(naked)]
extern "C" fn ok() {
    naked_asm!("")
}

impl S {
    #[unsafe(naked)]
    extern "C" fn g() {
        naked_asm!("")
    }

    #[unsafe(naked)]
    extern "C" fn h(&self) {
        naked_asm!("")
    }
}

fn main() {
    #[unsafe(naked)] //~ ERROR attribute cannot be used on
    || {};
}

// Check that the path of an attribute without a name is printed correctly (issue #140082)
#[::a]
//~^ ERROR attribute incompatible with `#[unsafe(naked)]`
//[edition2015]~| ERROR failed to resolve: use of unresolved module or unlinked crate `a`
//[edition2018]~| ERROR failed to resolve: could not find `a` in the list of imported crates
#[unsafe(naked)]
extern "C" fn issue_140082() {
    naked_asm!("")
}
