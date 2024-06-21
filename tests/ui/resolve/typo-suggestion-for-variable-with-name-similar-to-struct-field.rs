struct A {
    config: String,
}

impl A {
    fn new(cofig: String) -> Self {
        Self { config } //~ Error cannot find value `config`
    }

    fn do_something(cofig: String) {
        println!("{config}"); //~ Error cannot find value `config`
    }

    fn self_is_available(self, cofig: String) {
        println!("{config}"); //~ Error cannot find value `config`
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
        //~^ ERROR cannot find value `BAR`
        let foo: Baz = "".to_string();
        //~^ ERROR cannot find type `Baz`
    }
}

fn ba() {}
const BARR: u32 = 3;
type Bar = String;

fn main() {}
