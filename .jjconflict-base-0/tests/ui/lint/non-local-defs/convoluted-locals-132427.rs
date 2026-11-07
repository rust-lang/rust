// Regression tests for https://github.com/rust-lang/rust/issues/132427

//@ check-pass

// original
mod auth {
    const _: () = {
        pub enum ArbitraryContext {}

        const _: () = {
            impl ArbitraryContext {}
        };
    };
}

mod z {
    pub enum ArbitraryContext {}

    const _: () = {
        const _: () = {
            impl ArbitraryContext {}
        };
    };
}

const _: () = {
    mod auth {
        const _: () = {
            pub enum ArbitraryContext {}

            const _: () = {
                impl ArbitraryContext {}
            };
        };
    }
};

mod a {
    mod b {
        const _: () = {
            pub enum ArbitraryContext {}

            const _: () = {
                impl ArbitraryContext {}
            };
        };
    }
}

mod foo {
    const _: () = {
        mod auth {
            const _: () = {
                pub enum ArbitraryContext {}

                const _: () = {
                    impl ArbitraryContext {}
                };
            };
        }
    };
}

fn main() {}
