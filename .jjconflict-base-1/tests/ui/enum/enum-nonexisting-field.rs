//! regression test for <https://github.com/rust-lang/rust/issues/19922>

enum Homura {
    Akemi { madoka: () }
}

fn main() {
    let homura = Homura::Akemi { kaname: () };
    //~^ ERROR variant `Homura::Akemi` has no field named `kaname`
}
