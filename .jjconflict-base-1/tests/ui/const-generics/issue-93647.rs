struct X<const N: usize = {
    (||1usize)()
    //~^ ERROR cannot call non-const closure
}>;

fn main() {}
