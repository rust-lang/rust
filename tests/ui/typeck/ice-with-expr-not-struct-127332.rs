// Regression test for ICE #127332

// Tests that we do not ICE when a with expr is
// not a struct but something else like an enum

fn main() {
    let x = || {
        enum Foo {
            A { x: u32 },
        }
        let orig = Foo::A { x: 5 };
        Foo::A { x: 6, ..orig };
        //~^ ERROR functional record update syntax requires a struct
    };
}
