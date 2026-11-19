#![feature(doc_notable_trait)]

pub struct Wrapper<T> {
    inner: T,
}

impl<T: SomeTrait> SomeTrait for Wrapper<T> {}

#[doc(notable_trait)]
pub trait SomeTrait {
    //@ has doc_notable_trait/trait.SomeTrait.html
    //@ has - '//span[@class="tooltip"]/@data-notable-ty' 'Wrapper<Self>'
    //@ has doc_notable_trait/trait.SomeTrait.notable-traits-data.js
    //@ snapshot wrap-me -
    fn wrap_me(self) -> Wrapper<Self> where Self: Sized {
        Wrapper {
            inner: self,
        }
    }
}

pub struct SomeStruct;
impl SomeTrait for SomeStruct {}

impl SomeStruct {
    //@ has doc_notable_trait/struct.SomeStruct.html
    //@ has - '//span[@class="tooltip"]/@data-notable-ty' 'SomeStruct'
    //@ has doc_notable_trait/struct.SomeStruct.notable-traits-data.js
    //@ snapshot some-struct-new -
    pub fn new() -> SomeStruct {
        SomeStruct
    }
}

//@ has doc_notable_trait/fn.bare_fn.html
//@ has - '//span[@class="tooltip"]/@data-notable-ty' 'SomeStruct'
    //@ has doc_notable_trait/fn.bare_fn.notable-traits-data.js
//@ snapshot bare-fn -
pub fn bare_fn() -> SomeStruct {
    SomeStruct
}
