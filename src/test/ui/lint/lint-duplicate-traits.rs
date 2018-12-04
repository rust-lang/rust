// compile-pass

#![feature(trait_alias)]

trait Foo {}

trait SyncAlias = Sync;

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}

impl Foo for dyn Send + Sync + Send + SyncAlias {}

fn main() {}
