// Regression test for #15477. This test just needs to compile.

// pretty-expanded FIXME #23616

trait Chromosome<X: Chromosome<i32>> {
}

impl Chromosome<i32> for i32 { }

fn main() { }
