#[derive(Clone,
         Sync, //~ ERROR this unsafe trait should be implemented explicitly
         Copy)]
enum Foo {}
