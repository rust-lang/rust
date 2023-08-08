#![feature(const_fmt_arguments_new)]
#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    fn req(&self);

    fn prov(&self) {
        println!("lul"); //~ ERROR: cannot call non-const fn `_print` in constant functions
        self.req();
    }
}

struct S;

impl const Tr for S {
    fn req(&self) {}
}

fn main() {}
