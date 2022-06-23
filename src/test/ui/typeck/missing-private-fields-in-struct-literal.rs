pub mod m {
    pub struct S {
        pub visible: bool,
        a: (),
        b: (),
        c: (),
        d: (),
        e: (),
    }
}

fn main() {
    let _ = m::S { //~ ERROR cannot construct `S` with struct literal syntax due to private fields
        visible: true,
        a: (),
        b: (),
    };
}
