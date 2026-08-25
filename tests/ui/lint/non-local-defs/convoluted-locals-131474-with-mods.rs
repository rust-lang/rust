// This test check that no matter the nesting of const-anons and modules
// we consider them as transparent.
//
// Similar to https://github.com/rust-lang/rust/issues/131474

//@ check-pass

pub mod tmp {
    pub mod tmp {
        pub struct Test;
    }
}

const _: () = {
    const _: () = {
        const _: () = {
            const _: () = {
                impl tmp::tmp::Test {}
            };
        };
    };
};

const _: () = {
    const _: () = {
        mod tmp {
            pub(super) struct InnerTest;
        }

        impl tmp::InnerTest {}
    };
};

// https://github.com/rust-lang/rust/issues/131643
const _: () = {
    const _: () = {
        impl tmp::InnerTest {}
    };

    mod tmp {
        pub(super) struct InnerTest;
    }
};

fn main() {}
