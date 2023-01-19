// Unit test for the "user substitutions" that are annotated on each
// node.

// compile-flags:-Zverbose

#![feature(rustc_attrs)]

// Note: we reference the names T and U in the comments below.
trait Bazoom<T> {
    fn method<U>(&self, arg: T, arg2: U) {}
}

impl<S, T> Bazoom<T> for S {}

fn foo<'a, T>(_: T) {}

#[rustc_dump_user_substs]
fn main() {
    // Here: nothing is given, so we don't have any annotation.
    let x = foo;
    x(22);

    // Here: `u32` is given, which doesn't contain any lifetimes, so we don't
    // have any annotation.
    let x = foo::<u32>;
    x(22);

    let x = foo::<&'static u32>; //~ ERROR UserSubsts { substs: [Ref(ReStatic, Uint(U32), Not)], user_self_ty: None }
    x(&22);

    // Here: we only want the `T` to be given, the rest should be variables.
    //
    // (`T` refers to the declaration of `Bazoom`)
    let x = <_ as Bazoom<u32>>::method::<_>; //~ ERROR UserSubsts { substs: [Bound(DebruijnIndex(0), BoundTy { var: 0, kind: Anon }), Uint(U32), Bound(DebruijnIndex(0), BoundTy { var: 1, kind: Anon })], user_self_ty: None }
    x(&22, 44, 66);

    // Here: all are given and definitely contain no lifetimes, so we
    // don't have any annotation.
    let x = <u8 as Bazoom<u16>>::method::<u32>;
    x(&22, 44, 66);

    // Here: all are given and we have a lifetime.
    let x = <u8 as Bazoom<&'static u16>>::method::<u32>; //~ ERROR UserSubsts { substs: [Uint(U8), Ref(ReStatic, Uint(U16), Not), Uint(U32)], user_self_ty: None }
    x(&22, &44, 66);

    // Here: we want in particular that *only* the method `U`
    // annotation is given, the rest are variables.
    //
    // (`U` refers to the declaration of `Bazoom`)
    let y = 22_u32;
    y.method::<u32>(44, 66); //~ ERROR UserSubsts { substs: [Bound(DebruijnIndex(0), BoundTy { var: 0, kind: Anon }), Bound(DebruijnIndex(0), BoundTy { var: 1, kind: Anon }), Uint(U32)], user_self_ty: None }

    // Here: nothing is given, so we don't have any annotation.
    let y = 22_u32;
    y.method(44, 66);
}
