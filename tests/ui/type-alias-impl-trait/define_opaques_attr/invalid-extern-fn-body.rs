#![feature(type_alias_impl_trait)]

extern "C" {
    fn a() {
        //~^ ERROR incorrect function inside `extern` block
        #[define_opaque(String)]
        fn c() {}
    }
}

pub fn main() {}
