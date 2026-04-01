const fn test_me<T>(a: usize) -> usize {
    const { a }
    //~^ ERROR:  attempt to use a non-constant value in a constant
}

fn main() {}
