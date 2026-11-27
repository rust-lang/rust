// Test a supertrait cycle where the first trait we find (`A`) is not
// a direct participant in the cycle.
//@ ignore-parallel-frontend query cycle
trait A: B {
}

trait B: C {
    //~^ ERROR cycle detected
}

trait C: B { }

fn main() { }
