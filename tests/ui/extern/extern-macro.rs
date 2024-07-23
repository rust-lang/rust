// #41719

fn main() {
    enum Foo {}
    let _ = Foo::bar!(); //~ ERROR cannot find macro `bar`
}
