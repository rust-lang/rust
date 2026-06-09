// Make sure speculative path resolution works properly when resolution
// adjustment happens and no extra errors is reported.

struct S {
    field: u8,
}

trait Tr {
    fn method(&self);
}

impl Tr for S {
    fn method(&self) {
        fn g() {
            // Speculative resolution of `Self` and `self` silently fails,
            // "did you mean" messages are not printed.
            field;
            //~^ ERROR cannot find value `field`
            method();
            //~^ ERROR cannot find function `method`
        }

        field;
        //~^ ERROR cannot find value `field`
        method();
        //~^ ERROR cannot find function `method`
    }
}

fn main() {}
