fn main() {
    enum Foo { A { x: u32 } }
    let orig = Foo::A { x: 5 };
    Foo::A { x: 6, ..orig };
    //~^ ERROR functional record update syntax requires a struct
}
