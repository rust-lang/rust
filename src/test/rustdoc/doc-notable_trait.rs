#![feature(doc_notable_trait)]

pub struct Wrapper<T> {
    inner: T,
}

impl<T: SomeTrait> SomeTrait for Wrapper<T> {}

#[doc(notable_trait)]
pub trait SomeTrait {
    // @has doc_notable_trait/trait.SomeTrait.html
    // @has - '//code[@class="content"]' 'impl<T: SomeTrait> SomeTrait for Wrapper<T>'
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
    // @has - '//code[@class="content"]' 'impl SomeTrait for SomeStruct'
    // @has - '//code[@class="content"]' 'impl<T: SomeTrait> SomeTrait for Wrapper<T>'
    pub fn new() -> SomeStruct {
        SomeStruct
    }
}

// @has doc_notable_trait/fn.bare_fn.html
// @has - '//code[@class="content"]' 'impl SomeTrait for SomeStruct'
pub fn bare_fn() -> SomeStruct {
    SomeStruct
}
