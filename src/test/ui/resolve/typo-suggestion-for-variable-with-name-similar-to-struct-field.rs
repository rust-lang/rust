struct Foo {
    config: String,
}

impl Foo {
    fn new(cofig: String) -> Self {
        Self { config } //~ Error cannot find value `config` in this scope
    }

    fn do_something(cofig: String) {
        println!("{config}"); //~ Error cannot find value `config` in this scope
    }

    fn self_is_available(self, cofig: String) {
        println!("{config}"); //~ Error cannot find value `config` in this scope
    }
}

fn main() {}
