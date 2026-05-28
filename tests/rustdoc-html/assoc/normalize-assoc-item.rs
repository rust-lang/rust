// ignore-tidy-linelength
//@ aux-build:normalize-assoc-item.rs
//@ build-aux-docs
//@ compile-flags:-Znormalize-docs

pub trait Trait {
    type X;
}

impl Trait for usize {
    type X = isize;
}

impl Trait for () {
    type X = fn() -> i32;
}

impl Trait for isize {
    type X = <() as Trait>::X;
}

//@ has 'normalize_assoc_item/fn.f.html' '//pre[@class="rust item-decl"]' 'pub fn f() -> isize'
pub fn f() -> <usize as Trait>::X {
    0
}

//@ has 'normalize_assoc_item/fn.f2.html' '//pre[@class="rust item-decl"]' 'pub fn f2() -> fn() -> i32'
pub fn f2() -> <isize as Trait>::X {
    todo!()
}

pub struct S {
    //@ has 'normalize_assoc_item/struct.S.html' '//span[@id="structfield.box_me_up"]' 'box_me_up: Box<S>'
    pub box_me_up: <S as Trait>::X,
    //@ has 'normalize_assoc_item/struct.S.html' '//span[@id="structfield.generic"]' 'generic: (usize, isize)'
    pub generic: <Generic<usize> as Trait>::X,
}

impl Trait for S {
    type X = Box<S>;
}

pub struct Generic<Inner>(Inner);

impl<Inner: Trait> Trait for Generic<Inner> {
    type X = (Inner, Inner::X);
}

// These can't be normalized because they depend on a generic parameter.
// However the user can choose whether the text should be displayed as `Inner::X` or `<Inner as Trait>::X`.

//@ has 'normalize_assoc_item/struct.Unknown.html' '//pre[@class="rust item-decl"]' 'pub struct Unknown<Inner: Trait>(pub <Inner as Trait>::X);'
pub struct Unknown<Inner: Trait>(pub <Inner as Trait>::X);

//@ has 'normalize_assoc_item/struct.Unknown2.html' '//pre[@class="rust item-decl"]' 'pub struct Unknown2<Inner: Trait>(pub Inner::X);'
pub struct Unknown2<Inner: Trait>(pub Inner::X);

trait Lifetimes<'a> {
    type Y;
}

impl<'a> Lifetimes<'a> for usize {
    type Y = &'a isize;
}

//@ has 'normalize_assoc_item/fn.g.html' '//pre[@class="rust item-decl"]' "pub fn g() -> &'static isize"
pub fn g() -> <usize as Lifetimes<'static>>::Y {
    &0
}

//@ has 'normalize_assoc_item/constant.A.html' '//pre[@class="rust item-decl"]' "pub const A: &'static isize"
pub const A: <usize as Lifetimes<'static>>::Y = &0;

// test cross-crate re-exports
extern crate inner;
//@ has 'normalize_assoc_item/fn.foo.html' '//pre[@class="rust item-decl"]' "pub fn foo() -> i32"
pub use inner::foo;

//@ has 'normalize_assoc_item/fn.h.html' '//pre[@class="rust item-decl"]' "pub fn h<T>() -> IntoIter<T>"
pub fn h<T>() -> <Vec<T> as IntoIterator>::IntoIter {
    vec![].into_iter()
}
