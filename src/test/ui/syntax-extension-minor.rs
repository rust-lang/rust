#![feature(concat_idents)]

pub fn main() {
    struct Foo;
    let _: concat_idents!(F, oo) = Foo; // Test that `concat_idents!` can be used in type positions

    let asdf_fdsa = "<.<".to_string();
    // this now fails (correctly, I claim) because hygiene prevents
    // the assembled identifier from being a reference to the binding.
    assert!(concat_idents!(asd, f_f, dsa) == "<.<".to_string());
    //~^ ERROR cannot find value `asdf_fdsa` in this scope

    assert_eq!(stringify!(use_mention_distinction), "use_mention_distinction");
}
