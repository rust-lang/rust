// Test a supertrait cycle where a trait extends itself.

trait Chromosome: Chromosome {
    //~^ ERROR cycle detected
}

fn main() { }
