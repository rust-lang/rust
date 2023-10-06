// Make sure that we print the higher-ranked parameters of cross-crate function pointer types.
// They should be rendered exactly as the user wrote it, i.e., in source order and with unused
// parameters present, not stripped.

// aux-crate:fn_type=fn-type.rs
// edition: 2021
#![crate_name = "user"]

// @has user/type.F.html
// @has - '//*[@class="rust item-decl"]//code' \
//     "for<'z, 'a, '_unused> fn(_: &'z for<'b> fn(_: &'b str), _: &'a ()) -> &'a ();"
pub use fn_type::F;
