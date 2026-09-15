//@ check-pass
//! rustc_hir_analysis::check_item_type does type_of() on the default value. This is wonky, because
//! the generic args are in Self format at that point, not in impl format, so the result can't be
//! used with the Self-format args. However, it does not instantiate the result, it just does
//! ensure_ok(). This test just makes sure that codepath is hit in tests.
#![feature(min_generic_const_args, inherent_associated_types)]

struct Struct<T1, T2, T3>(T1, T2, T3);
impl<T1, T2, T3> Struct<T1, T2, T3> {
    const INHERENT: usize = core::direct_const_arg!(2);
}

struct WithDefault<const N: usize = { core::direct_const_arg!(Struct::<u8, u16, u32>::INHERENT) }>;

fn main() {}
