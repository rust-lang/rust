pub trait ThisTrait {}

mod asdf {
    use ThisTrait;

    pub struct SomeStruct;

    impl ThisTrait for SomeStruct {}

    trait PrivateTrait {}

    impl PrivateTrait for SomeStruct {}
}

// @has trait_vis/struct.SomeStruct.html
// @has - '//code' 'impl ThisTrait for SomeStruct'
// !@has - '//code' 'impl PrivateTrait for SomeStruct'
pub use asdf::SomeStruct;
