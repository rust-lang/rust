fn closure_ret_closure<T: FnOnce() -> T>(f: T) -> T {
    //~^ NOTE a bound in `closure_ret_closure` requires that a closure return itself, which is not possible
    //~| NOTE this requires the closure to return itself
    f()
}

fn main() {
    closure_ret_closure(|| 4); //~ ERROR [E0271]
    //~^ NOTE expected closure, found integer
    //~| NOTE the expected closure
    //~| NOTE this closure would have to return itself
    //~| NOTE expected closure
}
