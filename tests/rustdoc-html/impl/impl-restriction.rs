#![crate_name = "c"]
#![feature(impl_restriction)]

//@ matches c/trait.Foo.html '//*[@class="stab impl_restriction"]' \
//      'This trait cannot be implemented outside c.$'
//@ has c/trait.Foo.html '//*[@class="stab impl_restriction"]//code' 'c'
pub impl(crate) trait Foo {}

pub mod inner {
    //@ matches c/inner/trait.Bar.html '//*[@class="stab impl_restriction"]' \
    //      'This trait cannot be implemented outside c.$'
    //@ has c/inner/trait.Bar.html '//*[@class="stab impl_restriction"]//code' 'c'
    pub impl(self) trait Bar {}
}
