#![feature(doc_notable_trait)]
#![crate_name="foo"]

pub struct Wrapper<T> {
    inner: T,
}

impl<T> SomeTrait for Wrapper<T> {
    type SomeType = T;
}

#[doc(notable_trait)]
pub trait SomeTrait {
    type SomeType;
}

// @has foo/fn.bare_fn.html
// @has - '//a[@class="tooltip"]/@data-notable-ty' 'Wrapper<()>'
// @has - '//a[@class="tooltip"]/@title' 'impl<T> SomeTrait for Wrapper<T> type SomeType = T'
// @snapshot bare_fn_tooltip - '//a[@class="tooltip"]'
// @snapshot bare_fn_data - '//script[@id="notable-traits-data"]'
pub fn bare_fn() -> Wrapper<()> {
    unimplemented();
}
