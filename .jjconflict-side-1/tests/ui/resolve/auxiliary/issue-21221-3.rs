// testing whether the lookup mechanism picks up types
// defined in the outside crate

#![crate_type="lib"]

pub mod outer {
    // should suggest this
    pub trait OuterTrait {}

    // should not suggest this since the module is private
    mod private_module {
        pub trait OuterTrait {}
    }

    // should not suggest since the trait is private
    pub mod public_module {
        trait OuterTrait {}
    }
}
