use foo::bar::{
    self //~ ERROR module `bar` is private
};
use foo::bar::{
    Bar //~ ERROR module `bar` is private
};

mod foo {
    mod bar { pub type Bar = isize; }
}

fn main() {}
