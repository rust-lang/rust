//@ run-rustfix

fn add_ten<N>(n: N) -> N {
    n + 10
    //~^ ERROR cannot add `{integer}` to `N`
}

fn main() { add_ten(0); }
