// check-pass
//
// Makes sure we don't ICE when token capturing is triggered
// in an usual way

macro_rules! as_item {
    ($($i:item)*) => {$($i)*};
}

macro_rules! concat_attrs {
    ($($attrs:tt)*) => {
        as_item! {
            $($attrs)*
            pub struct Foo;
        }
    }
}

concat_attrs! { #[cfg(not(FALSE))] }


fn main() {}
