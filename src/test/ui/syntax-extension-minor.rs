// run-pass

#![feature(concat_idents)]

pub fn main() {
    struct Foo;
    let _: concat_idents!(F, oo) = Foo; // Test that `concat_idents!` can be used in type positions

    let asdf_fdsa = "<.<".to_string();
    // concat_idents should have call-site hygiene.
    assert!(concat_idents!(asd, f_f, dsa) == "<.<".to_string());

    assert_eq!(stringify!(use_mention_distinction), "use_mention_distinction");
}
