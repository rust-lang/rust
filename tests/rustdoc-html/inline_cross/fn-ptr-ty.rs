// Make sure that we print the higher-ranked parameters of cross-crate function pointer types.
// They should be rendered exactly as the user wrote it, i.e., in source order and with unused
// parameters present, not stripped.

//@ aux-crate:fn_ptr_ty=fn-ptr-ty.rs
//@ edition: 2021
#![crate_name = "user"]

//@ has user/type.F.html
//@ has - '//*[@class="rust item-decl"]//code' \
//     "for<'z, 'a, '_unused> fn(&'z for<'b> fn(&'b str), &'a ()) -> &'a ();"
pub use fn_ptr_ty::F;
