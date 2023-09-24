// Make sure specialization cannot change impl polarity

#![feature(rustc_attrs)]
#![feature(negative_impls)]
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

#[rustc_auto_trait]
trait Foo {}

impl<T> Foo for T {}
impl !Foo for u8 {} //~ ERROR E0751

#[rustc_auto_trait]
trait Bar {}

impl<T> !Bar for T {}
impl Bar for u8 {} //~ ERROR E0751

fn main() {}
