// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Print)]
#[allow(unused)]
enum ProceduralMasqueradeDummyType {
//~^ ERROR using
//~| WARN this was previously
    Input = (0, stringify!(input tokens!?)).0
}

fn main() {}
