// from rfc2005 test suite



// Without caching type lookups in FnCtxt.resolve_ty_and_def_ufcs
// the error below would be reported twice (once when checking
// for a non-ref pattern, once when processing the pattern).

fn main() {
    let foo = 22;
    let u32::XXX = foo else { return }; //~ ERROR: no associated item named `XXX` found for type `u32` in the current scope [E0599]
}
