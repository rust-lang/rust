#![crate_name = "foo"]

// @has foo/trait.LendingIterator.html
pub trait LendingIterator {
    // @has - '//*[@id="associatedtype.Item"]//h4[@class="code-header"]' "type Item<'a>where Self: 'a"
    type Item<'a> where Self: 'a;

    // @has - '//*[@id="tymethod.next"]//h4[@class="code-header"]' \
    //      "fn next<'a>(&'a self) -> Self::Item<'a>"
    // @has - '//*[@id="tymethod.next"]//h4[@class="code-header"]//a[@href="trait.LendingIterator.html#associatedtype.Item"]' \
    //      "Item"
    fn next<'a>(&'a self) -> Self::Item<'a>;
}

// @has foo/trait.LendingIterator.html
// @has - '//*[@id="associatedtype.Item-1"]//h4[@class="code-header"]' "type Item<'a> = ()"
impl LendingIterator for () {
    type Item<'a> = ();

    fn next<'a>(&self) -> () {}
}

pub struct Infinite<T>(T);

// @has foo/trait.LendingIterator.html
// @has - '//*[@id="associatedtype.Item-2"]//h4[@class="code-header"]' "type Item<'a>where Self: 'a = &'a T"
impl<T> LendingIterator for Infinite<T> {
    type Item<'a> where Self: 'a = &'a T;

    fn next<'a>(&'a self) -> Self::Item<'a> {
        &self.0
    }
}
