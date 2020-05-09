// check-pass
// compile-flags: -Z chalk

trait Foo {}

struct Bar {}

impl Foo for Bar {}

fn main() -> () {
    let _ = Bar {};
}
