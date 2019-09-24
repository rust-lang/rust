#[derive(Clone,
         Sync, //~ ERROR cannot find derive macro `Sync` in this scope
         Copy)]
enum Foo {}

fn main() {}
