#![feature(generic_const_exprs)]

// library portion of testing that `impl Trait<{ expr }>` doesnt
// ice because of a `DefKind::TyParam` parent
pub fn foo<const N: usize>(foo: impl Into<[(); N + 1]>) {
    foo.into();
}
