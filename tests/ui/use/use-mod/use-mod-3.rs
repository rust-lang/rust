use foo::bar::{ //~ ERROR module `bar` is private
    self
};
use foo::bar::{ //~ ERROR module `bar` is private
    Bar
};

mod foo {
    mod bar { pub type Bar = isize; }
}

fn main() {}
