crate struct Bender { //~ ERROR `crate` visibility modifier is experimental
    earth: bool,
    fire: bool,
    air: bool,
    water: bool,
}

macro_rules! accept_vis { ($v:vis) => {} }
accept_vis!(crate);  //~ ERROR `crate` visibility modifier is experimental

fn main() {}
