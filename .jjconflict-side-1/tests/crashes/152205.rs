//@ known-bug: #152205
#![deny(rust_2021_incompatible_closure_captures)]
struct Foo;
struct S;
impl Drop for S {
    fn drop(&mut self) {}
}
struct U(<Foo as NewTrait>::Assoc);
fn test_precise_analysis_long_path(u: U) {
    let _ = || {
        let _x = u.0.0;
    };
}
trait NewTrait {
    type Assoc;
}
impl NewTrait for Foo {
    type Assoc = (S,);
}
fn main(){}
