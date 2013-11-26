
trait Foo {}

fn take(f: ||:Foo) {
    //~^ ERROR only the builtin traits can be used as closure or object bounds
}

fn main() {}
