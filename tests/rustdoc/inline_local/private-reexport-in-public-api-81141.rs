// This test ensures that if a private re-export is present in a public API, it'll be
// replaced by the first public item in the re-export chain or by the private item.

// https://github.com/rust-lang/rust/issues/81141
#![crate_name = "foo"]

use crate::bar::Bar as Alias;

pub use crate::bar::Bar as Whatever;
use crate::Whatever as Whatever2;
use crate::Whatever2 as Whatever3;
pub use crate::bar::Inner as Whatever4;

mod bar {
    pub struct Bar;
    pub use self::Bar as Inner;
}

//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar() -> Bar'
pub fn bar() -> Alias {
    Alias
}

//@ has 'foo/fn.bar2.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar2() -> Whatever'
pub fn bar2() -> Whatever3 {
    Whatever
}

//@ has 'foo/fn.bar3.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar3() -> Whatever4'
pub fn bar3() -> Whatever4 {
    Whatever
}

//@ has 'foo/fn.bar4.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar4() -> Bar'
pub fn bar4() -> crate::Alias {
    Alias
}

//@ has 'foo/fn.bar5.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar5() -> Whatever'
pub fn bar5() -> crate::Whatever3 {
    Whatever
}

//@ has 'foo/fn.bar6.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar6() -> Whatever4'
pub fn bar6() -> crate::Whatever4 {
    Whatever
}


//@ has 'foo/fn.bar7.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar7() -> Bar'
pub fn bar7() -> self::Alias {
    Alias
}

//@ has 'foo/fn.bar8.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar8() -> Whatever'
pub fn bar8() -> self::Whatever3 {
    Whatever
}

//@ has 'foo/fn.bar9.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar9() -> Whatever4'
pub fn bar9() -> self::Whatever4 {
    Whatever
}

mod nested {
    pub(crate) use crate::Alias;
    pub(crate) use crate::Whatever3;
    pub(crate) use crate::Whatever4;
    pub(crate) use crate::nested as nested2;
}

//@ has 'foo/fn.bar10.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar10() -> Bar'
pub fn bar10() -> nested::Alias {
    Alias
}

//@ has 'foo/fn.bar11.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar11() -> Whatever'
pub fn bar11() -> nested::Whatever3 {
    Whatever
}

//@ has 'foo/fn.bar12.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar12() -> Whatever4'
pub fn bar12() -> nested::Whatever4 {
    Whatever
}

//@ has 'foo/fn.bar13.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar13() -> Bar'
pub fn bar13() -> nested::nested2::Alias {
    Alias
}

//@ has 'foo/fn.bar14.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar14() -> Whatever'
pub fn bar14() -> nested::nested2::Whatever3 {
    Whatever
}

//@ has 'foo/fn.bar15.html'
//@ has - '//*[@class="rust item-decl"]/code' 'pub fn bar15() -> Whatever4'
pub fn bar15() -> nested::nested2::Whatever4 {
    Whatever
}

use external::Public as Private;

pub mod external {
    pub struct Public;

    //@ has 'foo/external/fn.make.html'
    //@ has - '//*[@class="rust item-decl"]/code' 'pub fn make() -> Public'
    pub fn make() -> super::Private { super::Private }
}
