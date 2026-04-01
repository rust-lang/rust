struct S;

impl S {
    fn a(&self: Self) {}
    //~^ ERROR type not allowed for shorthand `self` parameter
    fn b(&mut self: Self) {}
    //~^ ERROR type not allowed for shorthand `self` parameter
    fn c<'c>(&'c mut self: Self) {}
    //~^ ERROR type not allowed for shorthand `self` parameter
    fn d<'d>(&'d self: Self) {}
    //~^ ERROR type not allowed for shorthand `self` parameter
}

fn main() {}
