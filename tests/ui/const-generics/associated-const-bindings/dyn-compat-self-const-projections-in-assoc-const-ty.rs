// FIXME(mgca): Ideally this would compile -- at least if the user annotated the instantiated type
//              of the assoc const (but we don't have the syntax for this (yet)). In any case, we
//              should not leak `trait_object_dummy_self` (defined as `FreshTy(0)` under the hood)
//              to the rest of the compiler and by extension the user via diagnostics.
//@ known-bug: unknown

#![feature(min_generic_const_args, unsized_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

trait A {
    type Ty: std::marker::ConstParamTy_;
    #[type_const] const CT: Self::Ty;
}

impl A for () {
    type Ty = i32;
    #[type_const] const CT: i32 = 0;
}

fn main() {
    // NOTE: As alluded to above, if we can't get the examples below to compile as written,
    //       we might want to allow the user to manually specify the instantiated type somehow.
    //       The hypothetical syntax for that *might* look sth. like
    //       * `dyn A<Ty = i32, CT = const -> i32 { 0 }>`
    //       * `dyn A<Ty = i32, CT: i32 = 0>`

    let _: dyn A<Ty = i32, CT = 0>;

    let _: &dyn A<Ty = i32, CT = 0> = &();
}
