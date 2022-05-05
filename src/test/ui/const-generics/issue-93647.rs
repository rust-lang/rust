struct X<const N: usize = {
    (||1usize)()
    //~^ ERROR cannot call
}>;

fn main() {}
