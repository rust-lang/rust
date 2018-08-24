extern {
    fn sqrt<T>(f: T) -> T;
    //~^ ERROR foreign items may not have type parameters [E0044]
    //~| HELP use specialization instead of type parameters by replacing them with concrete types
    //~| NOTE can't have type parameters
}

fn main() {
}
