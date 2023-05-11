// when implementing the fix for traits-in-bodies, there was an ICE when documenting private items
// and a trait was defined in non-module scope

// compile-flags:--document-private-items

// @has traits_in_bodies_private/struct.SomeStruct.html
// @!has - '//code' 'impl HiddenTrait for SomeStruct'
pub struct SomeStruct;

fn __implementation_details() {
    trait HiddenTrait {}
    impl HiddenTrait for SomeStruct {}
}
