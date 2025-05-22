//@ known-bug: #140333
fn a() -> impl b<
    [c; {
        struct d {
            #[a]
            bar: e,
        }
    }],
>;
