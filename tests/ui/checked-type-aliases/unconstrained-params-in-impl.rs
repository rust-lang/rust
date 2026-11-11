#![feature(checked_type_aliases)]
#![allow(incomplete_features)]

impl<T> NotInjective<T> {} //~ ERROR the type parameter `T` is not constrained

type NotInjective<T: ?Sized> = Local<<T as Discard>::Out>;
struct Local<T>(T);

trait Discard { type Out; }
impl<T: ?Sized> Discard for T { type Out = (); }

fn main() {}
