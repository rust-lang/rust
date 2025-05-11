//@[gate] check-pass
//@ revisions: gate nogate
#![cfg_attr(gate, feature(generic_arg_infer))]

struct Foo<const N: usize>;

fn main() {
    // AST Types preserve parens for pretty printing reasons. This means
    // that this is parsed as a `TyKind::Paren(TyKind::Infer)`. Generic
    // arg lowering therefore needs to take into account not just `TyKind::Infer`
    // but `TyKind::Infer` wrapped in arbitrarily many `TyKind::Paren`.
    let a: Vec<(_)> = vec![1_u8];
    let a: Vec<(((((_)))))> = vec![1_u8];

    // AST Exprs similarly preserve parens for pretty printing reasons.
    #[rustfmt::skip]
    let b: [u8; (_)] = [1; (((((_)))))];
    //[nogate]~^ error: using `_` for array lengths is unstable
    //[nogate]~| error: using `_` for array lengths is unstable
    let b: [u8; 2] = b;

    // This is the same case as AST types as the parser doesn't distinguish between const
    // and type args when they share syntax
    let c: Foo<_> = Foo::<1>;
    //[nogate]~^ error: const arguments cannot yet be inferred with `_`
    let c: Foo<(_)> = Foo::<1>;
    //[nogate]~^ error: const arguments cannot yet be inferred with `_`
    let c: Foo<(((_)))> = Foo::<1>;
    //[nogate]~^ error: const arguments cannot yet be inferred with `_`
}
