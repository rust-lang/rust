extern "C" {
    type Item = [T] where [T]: Sized;
    //~^ incorrect `type` inside `extern` block
    //~| `type`s inside `extern` blocks cannot have `where` clauses
    //~| cannot find type `T`
    //~| cannot find type `T`
    //~| extern types are experimental
}

fn main() {}
