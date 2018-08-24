// Without caching type lookups in FnCtxt.resolve_ty_and_def_ufcs
// the error below would be reported twice (once when checking
// for a non-ref pattern, once when processing the pattern).

fn main() {
    let foo = 22;
    match foo {
        u32::XXX => { } //~ ERROR no associated item named
        _ => { }
    }
}
