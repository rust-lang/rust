// ignore-tidy-linelength

pub struct Struct;

impl Struct {
    pub fn function(funs: Vec<Fn() -> ()>) {}
    //~^ ERROR the size for values of type
}

fn main() {}
