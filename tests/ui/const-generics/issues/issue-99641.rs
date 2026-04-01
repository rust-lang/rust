#![feature(adt_const_params)]
#![allow(incomplete_features)]

fn main() {
    pub struct Color<const WHITE: (fn(),)>;
    //~^ ERROR `(fn(),)` can't be used as a const parameter type

    impl<const WHITE: (fn(),)> Color<WHITE> {
        //~^ ERROR `(fn(),)` can't be used as a const parameter type
        pub fn new() -> Self {
            Color::<WHITE>
        }
    }

    pub const D65: (fn(),) = (|| {},);

    Color::<D65>::new();
}
