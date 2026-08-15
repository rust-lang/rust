#![feature(type_alias_impl_trait)]

// This test ensures that unnameable types stay unnameable
// https://github.com/rust-lang/rust/issues/63063#issuecomment-1360053614

// library
mod private {
    pub struct Private;
    pub trait Trait {
        fn dont_define_this(_private: Private) {}
    }
}

use private::Trait;

// downstream
type MyPrivate = impl Sized;
impl Trait for u32 {
    #[define_opaque(MyPrivate)]
    fn dont_define_this(private: MyPrivate) {
        //~^ ERROR: incompatible type for trait
        let _: () = private;
    }
}

fn main() {}
