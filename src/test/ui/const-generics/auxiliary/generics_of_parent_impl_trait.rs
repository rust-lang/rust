#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

// library portion of testing that `impl Trait<{ expr }>` doesnt
// ice because of a `DefKind::TyParam` parent
pub fn foo<const N: usize>(foo: impl Into<[(); N + 1]>) {
    foo.into();
}
