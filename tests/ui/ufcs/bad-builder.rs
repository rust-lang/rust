fn hello<Q>() -> Vec<Q> {
    Vec::<Q>::mew()
    //~^ ERROR no associated function or constant named `mew` found for struct `Vec<Q>` in the current scope
}

fn main() {}
