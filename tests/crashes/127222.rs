//@ known-bug: rust-lang/rust#127222
#[marker]
trait Foo = PartialEq<i32> + Send;
