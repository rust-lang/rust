struct A {
    config: String,
}

impl A {
    fn new(cofig: String) -> Self {
        Self { config } //~ ERROR cannot find value `config` in this scope
    }

    fn do_something(cofig: String) {
        println!("{config}"); //~ ERROR cannot find value `config` in this scope
    }

    fn self_is_available(self, cofig: String) {
        println!("{config}"); //~ ERROR cannot find value `config` in this scope
    }
}

trait B {
    const BAR: u32 = 3;
    type Baz;
    fn bar(&self);
    fn baz(&self) {}
    fn bah() {}
}

impl B for Box<isize> {
    type Baz = String;
    fn bar(&self) {
        // let baz = 3;
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
        BAR;
        //~^ ERROR cannot find value `BAR` in this scope
        let foo: Baz = "".to_string();
        //~^ ERROR cannot find type `Baz` in this scope
    }
}

fn ba() {}
const BARR: u32 = 3;
type Bar = String;

fn main() {}
