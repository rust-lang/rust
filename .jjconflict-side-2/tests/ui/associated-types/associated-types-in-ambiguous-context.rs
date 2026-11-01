//@ normalize-stderr: "and \d+ other candidates" -> "and N other candidates"

trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

fn get<T:Get,U:Get>(x: T, y: U) -> Get::Value {}
//~^ ERROR ambiguous associated type

trait Grab {
    type Value;
    fn grab(&self) -> Grab::Value;
    //~^ ERROR ambiguous associated type

    fn get(&self) -> Get::Value;
    //~^ ERROR ambiguous associated type
}

trait Bar {}

trait Foo where Foo::Assoc: Bar {
//~^ ERROR ambiguous associated type
    type Assoc;
}

type X = std::ops::Deref::Target;
//~^ ERROR ambiguous associated type

fn main() {
}
