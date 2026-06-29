// Checks that links to primitive types methods work.
// Regression test for <https://github.com/rust-lang/rust/issues/156707>.

// ignore-tidy-linelength
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/prim-method.rs.html'

fn scope() {
    //@ has - '//a[@href="{{channel}}/core/primitive.usize.html#method.saturating_add"]' 'saturating_add'
    let _ = 0usize.saturating_add(1);
    //@ has - '//a[@href="{{channel}}/core/primitive.bool.html#method.then_some"]' 'then_some'
    let _ = false.then_some(());
}
