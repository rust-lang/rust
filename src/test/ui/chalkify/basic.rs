// check-pass
// compile-flags: -Z trait-solver=chalk

trait Foo {}

struct Bar {}

impl Foo for Bar {}

fn main() -> () {
    let _ = Bar {};
}
