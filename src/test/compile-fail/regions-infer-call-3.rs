fn select(x: &r/int, y: &r/int) -> &r/int { x }

fn with<T>(f: fn(x: &int) -> T) -> T {
    f(&20)
}

fn manip(x: &a/int) -> int {
    let z = do with |y| { select(x, y) };
    //~^ ERROR cannot infer an appropriate lifetime
    *z
}

fn main() {
}