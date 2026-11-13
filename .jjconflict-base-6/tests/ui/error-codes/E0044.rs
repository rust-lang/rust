extern "C" {
    fn sqrt<T>(f: T) -> T;
//~^ ERROR foreign items may not have type parameters [E0044]
//~| HELP replace the type parameters with concrete types
//~| NOTE can't have type parameters
}

fn main() {}
