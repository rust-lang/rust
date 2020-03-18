fn foo<const A: usize, const B: usize>(bar: [usize; A + B]) {
    //~^ ERROR const generics are unstable
}

fn main() { }
