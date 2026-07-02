//@ check-pass

#![feature(min_adt_const_params)]
#![feature(min_generic_const_args)]

struct S<const X: (u32, u32)>;

fn main() {
    let _: S<{ (1, _) }> = S::<{ (1, 2) }>;
}
