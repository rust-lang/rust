#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

trait Tr {
    fn req(&self);

    fn prov(&self) {
        println!("lul");
        self.req();
    }

    #[default_method_body_is_const]
    fn default() {}
}

struct S;

impl const Tr for S {
    fn req(&self) {}
} //~^^ ERROR const trait implementations may not use non-const default functions

impl const Tr for u16 {
    fn prov(&self) {}
    fn default() {}
} //~^^^ ERROR not all trait items implemented


impl const Tr for u32 {
    fn req(&self) {}
    fn default() {}
} //~^^^ ERROR const trait implementations may not use non-const default functions

fn main() {}
