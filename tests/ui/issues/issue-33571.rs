#[derive(Clone,
         Sync, //~ ERROR cannot find derive macro `Sync`
               //~| ERROR cannot find derive macro `Sync`
         Copy)]
enum Foo {}

fn main() {}
