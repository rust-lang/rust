// #41719

fn main() {
    enum Foo {}
    let _ = Foo::bar!(); //~ ERROR failed to resolve: partially resolved path in a macro
}
