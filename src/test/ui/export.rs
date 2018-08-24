mod foo {
    pub fn x(y: isize) { log(debug, y); }
    //~^ ERROR cannot find function `log` in this scope
    //~| ERROR cannot find value `debug` in this scope
    fn z(y: isize) { log(debug, y); }
    //~^ ERROR cannot find function `log` in this scope
    //~| ERROR cannot find value `debug` in this scope
}

fn main() { foo::z(10); } //~ ERROR function `z` is private
