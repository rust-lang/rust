#![feature(impl_trait_in_assoc_type)]

pub struct AlwaysTrue;

//@ has impl_trait_in_assoc_type/struct.AlwaysTrue.html

impl IntoIterator for AlwaysTrue {
    type Item = bool;

    //@ has - '//*[@id="associatedtype.IntoIter"]//h4[@class="code-header"]' \
    //  'type IntoIter = impl Iterator<Item = bool>'
    type IntoIter = impl Iterator<Item = bool>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat(true)
    }
}
