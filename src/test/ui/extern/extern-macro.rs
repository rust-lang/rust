// #41719

fn main() {
    enum Foo {}
    let _ = Foo::bar!(); //~ ERROR fail to resolve non-ident macro path
}
