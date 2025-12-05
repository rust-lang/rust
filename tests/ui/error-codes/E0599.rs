struct Foo;

fn main() {
    || if let Foo::NotEvenReal() = Foo {}; //~ ERROR E0599
}
