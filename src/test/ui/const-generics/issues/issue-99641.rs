#![feature(adt_const_params)]
#![allow(incomplete_features)]

fn main() {
    pub struct Color<const WHITE: (fn(),)>;
    //~^ ERROR using function pointers

    impl<const WHITE: (fn(),)> Color<WHITE> {
        //~^ ERROR using function pointers
        pub fn new() -> Self {
            Color::<WHITE>
        }
    }

    pub const D65: (fn(),) = (|| {},);

    Color::<D65>::new();
}
