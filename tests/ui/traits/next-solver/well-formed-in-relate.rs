fn main() {
    let x;
    //~^ ERROR type annotations needed for `Map<_, _>`
    higher_ranked();
    x = unconstrained_map();
}

fn higher_ranked() where for<'a> &'a (): Sized {}

struct Map<T, U> where T: Fn() -> U {
    t: T,
}

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

fn unconstrained_map<T: Fn() -> U, U>() -> <Map<T, U> as Mirror>::Assoc { todo!() }
