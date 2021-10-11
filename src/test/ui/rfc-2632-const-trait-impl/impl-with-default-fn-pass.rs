// check-pass

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

impl const Tr for u8 {
    fn req(&self) {}
    fn prov(&self) {}
}

macro_rules! impl_tr {
    ($ty: ty) => {
        impl const Tr for $ty {
            fn req(&self) {}
            fn prov(&self) {}
        }
    }
}

impl_tr!(u64);

fn main() {}
