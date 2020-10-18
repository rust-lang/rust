// ignore-tidy-linelength
// aux-build:normalize-assoc-item.rs
// build-aux-docs

pub trait Trait {
    type X;
}

impl Trait for usize {
    type X = isize;
}

// @has 'normalize_assoc_item/fn.f.html' '//pre[@class="rust fn"]' 'pub fn f() -> isize'
pub fn f() -> <usize as Trait>::X {
    0
}

pub struct S {
    // @has 'normalize_assoc_item/struct.S.html' '//span[@id="structfield.box_me_up"]' 'box_me_up: Box<S, Global>'
    pub box_me_up: <S as Trait>::X,
    // @has 'normalize_assoc_item/struct.S.html' '//span[@id="structfield.generic"]' 'generic: (usize, isize)'
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

// @has 'normalize_assoc_item/struct.Unknown.html' '//pre[@class="rust struct"]' 'pub struct Unknown<Inner: Trait>(pub <Inner as Trait>::X);'
pub struct Unknown<Inner: Trait>(pub <Inner as Trait>::X);

// @has 'normalize_assoc_item/struct.Unknown2.html' '//pre[@class="rust struct"]' 'pub struct Unknown2<Inner: Trait>(pub Inner::X);'
pub struct Unknown2<Inner: Trait>(pub Inner::X);

trait Lifetimes<'a> {
    type Y;
}

impl<'a> Lifetimes<'a> for usize {
    type Y = &'a isize;
}

// @has 'normalize_assoc_item/fn.g.html' '//pre[@class="rust fn"]' "pub fn g() -> &isize"
pub fn g() -> <usize as Lifetimes<'static>>::Y {
    &0
}

// @has 'normalize_assoc_item/constant.A.html' '//pre[@class="rust const"]' "pub const A: &isize"
pub const A: <usize as Lifetimes<'static>>::Y = &0;

// test cross-crate re-exports
extern crate inner;
// @has 'normalize_assoc_item/fn.foo.html' '//pre[@class="rust fn"]' "pub fn foo() -> i32"
pub use inner::foo;

// @has 'normalize_assoc_item/fn.h.html' '//pre[@class="rust fn"]' "pub fn h<T>() -> IntoIter<T, Global>"
pub fn h<T>() -> <Vec<T> as IntoIterator>::IntoIter {
    vec![].into_iter()
}
