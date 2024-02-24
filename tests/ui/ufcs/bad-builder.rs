fn hello<Q>() -> Vec<Q> {
    Vec::<Q>::mew()
    //~^ ERROR no function or associated item named `mew` found for struct `Vec<Q>` in the current scope
}

fn main() {}
