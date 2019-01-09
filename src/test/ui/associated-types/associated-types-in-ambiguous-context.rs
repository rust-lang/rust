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
}

type X = std::ops::Deref::Target;
//~^ ERROR ambiguous associated type

fn main() {
}
