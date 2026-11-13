// This test check that no matter the nesting of const-anons we consider
// them as transparent.
//
// https://github.com/rust-lang/rust/issues/131474

//@ check-pass

pub struct Test;

const _: () = {
    const _: () = {
        impl Test {}
    };
};

const _: () = {
    const _: () = {
        struct InnerTest;

        impl InnerTest {}
    };
};

// https://github.com/rust-lang/rust/issues/131643
const _: () = {
    const _: () = {
        impl InnerTest {}
    };

    struct InnerTest;
};

fn main() {}
