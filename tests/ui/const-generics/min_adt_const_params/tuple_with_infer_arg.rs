//@ check-pass

#![feature(min_adt_const_params, gca_min_const_items, gca_macroless_args)]

struct S<const X: (u32, u32)>;

fn main() {
    let _: S<{ (1, _) }> = S::<{ (1, 2) }>;
}
