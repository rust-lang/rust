//@ has assoc_const_stringify/struct.Signed.html '//*[@class="docblock"]//code' 'i32::MAX'
//@ has assoc_const_stringify/struct.Unsigned.html '//*[@class="docblock"]//code' 'usize::MAX'

#![crate_name = "assoc_const_stringify"]

macro_rules! max_docs {
    ($name:ident, $ty:ident) => {
        pub struct $name;

        impl $name {
            #[doc = concat!("`", stringify!($ty::MAX), "`")]
            pub fn max() {}
        }
    };
}

max_docs!(Signed, i32);
max_docs!(Unsigned, usize);
