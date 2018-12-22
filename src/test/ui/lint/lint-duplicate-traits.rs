// compile-pass

#![feature(trait_alias)]

trait Foo {}
trait Bar {}
trait Baz {}

trait SyncAlias = Sync;

impl Foo for dyn Send {}

impl Bar for dyn Send + Send {}

impl Baz for dyn Send + Sync + Send + SyncAlias {}

fn main() {}
