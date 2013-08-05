
trait Foo {}

fn take(f: &fn:Foo()) {
    //~^ ERROR only the builtin traits can be used as closure or object bounds
}

fn main() {}
