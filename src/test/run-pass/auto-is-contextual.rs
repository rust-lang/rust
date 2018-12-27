#![allow(path_statements)]
#![allow(dead_code)]
macro_rules! auto {
    () => (struct S;)
}

auto!();

fn auto() {}

fn main() {
    auto();
    let auto = 10;
    auto;
    auto as u8;
}
