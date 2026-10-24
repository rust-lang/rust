//! regression test for <https://github.com/rust-lang/rust/issues/19692>

struct Homura;

fn akemi(homura: Homura) {
    let Some(ref madoka) = Some(homura.kaname()); //~ ERROR no method named `kaname` found
    madoka.clone();
}

fn main() { }
