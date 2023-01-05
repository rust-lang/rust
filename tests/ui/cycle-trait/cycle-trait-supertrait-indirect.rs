// Test a supertrait cycle where the first trait we find (`A`) is not
// a direct participant in the cycle.

trait A: B {
}

trait B: C {
    //~^ ERROR cycle detected
}

trait C: B { }

fn main() { }
