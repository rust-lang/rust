pub trait ThisTrait {}

mod asdf {
    use ThisTrait;

    pub struct SomeStruct;

    impl ThisTrait for SomeStruct {}

    trait PrivateTrait {}

    impl PrivateTrait for SomeStruct {}
}

//@ has trait_vis/struct.SomeStruct.html
//@ has - '//h3[@class="code-header"]' 'impl ThisTrait for SomeStruct'
//@ !has - '//h3[@class="code-header"]' 'impl PrivateTrait for SomeStruct'
pub use asdf::SomeStruct;
