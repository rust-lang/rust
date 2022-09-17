struct X<const N: usize = {
    (||1usize)()
    //~^ ERROR the trait bound
}>;

fn main() {}
