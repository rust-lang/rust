// This test checks that the --edition is respected by the crate-attr attributes

//@ revisions: e2021 e2024
//@ compile-flags: -Zcrate-attr=doc=stringify!(#"a"#)

//@[e2021] edition: 2021
//@[e2021] check-pass

//@[e2024] edition: 2024
//[e2024]~? ERROR invalid string literal

fn main() {}
