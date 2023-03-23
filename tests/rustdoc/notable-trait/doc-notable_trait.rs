#![feature(doc_notable_trait)]

pub struct Wrapper<T> {
    inner: T,
}

impl<T: SomeTrait> SomeTrait for Wrapper<T> {}

#[doc(notable_trait)]
pub trait SomeTrait {
    // @has doc_notable_trait/trait.SomeTrait.html
    // @has - '//a[@class="tooltip"]/@data-notable-ty' 'Wrapper<Self>'
    // @snapshot wrap-me - '//script[@id="notable-traits-data"]'
    fn wrap_me(self) -> Wrapper<Self> where Self: Sized {
        Wrapper {
            inner: self,
        }
    }
}

pub struct SomeStruct;
impl SomeTrait for SomeStruct {}

impl SomeStruct {
    // @has doc_notable_trait/struct.SomeStruct.html
    // @has - '//a[@class="tooltip"]/@data-notable-ty' 'SomeStruct'
    // @snapshot some-struct-new - '//script[@id="notable-traits-data"]'
    pub fn new() -> SomeStruct {
        SomeStruct
    }
}

// @has doc_notable_trait/fn.bare_fn.html
// @has - '//a[@class="tooltip"]/@data-notable-ty' 'SomeStruct'
// @snapshot bare-fn - '//script[@id="notable-traits-data"]'
pub fn bare_fn() -> SomeStruct {
    SomeStruct
}
