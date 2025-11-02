//@ known-bug: rust-lang/rust#143872
//@ needs-rustc-debug-assertions
trait Project {
    type Ty;
}
impl Project for &'_ &'static () {
    type Ty = ();
}
trait Trait {
    fn get<'s>(s: &'s str, _: ()) -> &'_ str;
}
impl Trait for () {
    fn get<'s>(s: &'s str, _: <&&'s () as Project>::Ty) -> &'static str {
        s
    }
}
