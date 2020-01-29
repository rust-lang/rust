#![deny(patterns_in_fns_without_body)]

trait Tr {
    fn f1(mut arg: u8); //~ ERROR patterns aren't allowed in functions without bodies
                        //~^ WARN was previously accepted
    fn f2(&arg: u8); //~ ERROR patterns aren't allowed in functions without bodies
    fn g1(arg: u8); // OK
    fn g2(_: u8); // OK
    #[allow(anonymous_parameters)]
    fn g3(u8); // OK
}

fn main() {}
