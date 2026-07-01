//@ edition:2015
// FIXME(fmease): Rename file name (e.g., `arg` -> `param`).
// FIXME(fmease): Add historical context.
#![deny(patterns_in_fns_without_body)]

trait Tr {
    fn f1(mut arg: u8); //~ ERROR patterns aren't allowed in functions without bodies
                        //~^ WARN was previously accepted
    fn f2(&arg: u8);
    //~^ ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
    fn g1(arg: u8); // OK
    fn g2(_: u8); // OK
    #[allow(anonymous_parameters)]
    fn g3(u8); // OK
}

fn main() {}
