// check-pass
// edition: 2021
// aux-crate:is_derive_const=is-derive-const.rs
#![feature(derive_const)]

const _: () = {
    #[derive(is_derive_const::IsDeriveConst)]
    struct _Type;

    assert!(!IS_DERIVE_CONST);
};

const _: () = {
    #[derive_const(is_derive_const::IsDeriveConst)]
    struct _Type;

    assert!(IS_DERIVE_CONST);
};

fn main() {}
