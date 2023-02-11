#![crate_type="lib"]

// @has assoc_types/trait.Index.html
pub trait Index<I: ?Sized> {
    // @has - '//*[@id="associatedtype.Output"]//h4[@class="code-header"]' 'type Output: ?Sized'
    type Output: ?Sized;
    // @has - '//*[@id="tymethod.index"]//h4[@class="code-header"]' \
    //      "fn index<'a>(&'a self, index: I) -> &'a Self::Output"
    // @has - '//*[@id="tymethod.index"]//h4[@class="code-header"]//a[@href="trait.Index.html#associatedtype.Output"]' \
    //      "Output"
    fn index<'a>(&'a self, index: I) -> &'a Self::Output;
}

// @has assoc_types/fn.use_output.html
// @has - '//pre[@class="rust item-decl"]' '-> &T::Output'
// @has - '//pre[@class="rust item-decl"]//a[@href="trait.Index.html#associatedtype.Output"]' 'Output'
pub fn use_output<T: Index<usize>>(obj: &T, index: usize) -> &T::Output {
    obj.index(index)
}

pub trait Feed {
    type Input;
}

// @has assoc_types/fn.use_input.html
// @has - '//pre[@class="rust item-decl"]' 'T::Input'
// @has - '//pre[@class="rust item-decl"]//a[@href="trait.Feed.html#associatedtype.Input"]' 'Input'
pub fn use_input<T: Feed>(_feed: &T, _element: T::Input) { }

// @has assoc_types/fn.cmp_input.html
// @has - '//pre[@class="rust item-decl"]' 'where T::Input: PartialEq<U::Input>'
// @has - '//pre[@class="rust item-decl"]//a[@href="trait.Feed.html#associatedtype.Input"]' 'Input'
pub fn cmp_input<T: Feed, U: Feed>(a: &T::Input, b: &U::Input) -> bool
    where T::Input: PartialEq<U::Input>
{
    a == b
}
