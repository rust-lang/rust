extern "C" {
    type Item = [T] where [T]: Sized;
    //~^ ERROR incorrect `type` inside `extern` block
    //~| ERROR `type`s inside `extern` blocks cannot have `where` clauses
    //~| ERROR cannot find type `T` in this scope
    //~| ERROR cannot find type `T` in this scope
    //~| ERROR extern types are experimental
}

fn main() {}
