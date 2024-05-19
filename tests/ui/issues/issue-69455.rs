// Regression test for #69455: projection predicate was not satisfied.
// Compiler should indicate the correct location of the
// unsatisfied projection predicate

pub trait Test<Rhs = Self> {
    type Output;

    fn test(self, rhs: Rhs) -> Self::Output;
}

impl Test<u32> for u64 {
    type Output = u64;

    fn test(self, other: u32) -> u64 {
        self + (other as u64)
    }
}

impl Test<u64> for u64 {
    type Output = u64;

    fn test(self, other: u64) -> u64 {
        (self + other) as u64
    }
}

fn main() {
    let xs: Vec<u64> = vec![1, 2, 3];
    println!("{}", 23u64.test(xs.iter().sum())); //~ ERROR: type annotations needed
    //~^ ERROR type annotations needed
}
