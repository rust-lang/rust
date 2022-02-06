struct X<const N: usize = {
    (||1usize)()
    //~^ ERROR calls in constants are limited to
}>;

fn main() {}
