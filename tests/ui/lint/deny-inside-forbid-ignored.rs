//! Ensure that using deny inside forbid is treated as a no-op, and does not override the level to
//! deny.

//@ revisions: source_only cli_forbid cli_forbid_warnings
//@[cli_forbid] compile-flags: -F unsafe_code
//@[cli_forbid_warnings] compile-flags: -F warnings

#[forbid(unsafe_code)] // NO UNSAFE CODE IN HERE!!
fn main() {
    #[deny(unsafe_code)] // m-m-maybe we can have unsafe code in here?
    {
        #[allow(unsafe_code)] // let's have some unsafe code in here
        //~^ ERROR allow(unsafe_code) incompatible with previous forbid
        //~| ERROR allow(unsafe_code) incompatible with previous forbid
        {
            unsafe { /* ≽^•⩊•^≼ */ }
            //~^ ERROR usage of an `unsafe` block
        }
    }
}
