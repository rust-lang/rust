pub struct Struct;

impl Struct {
    pub fn function(funs: Vec<dyn Fn() -> ()>) {}
    //~^ ERROR the size for values of type
}

fn main() {}
