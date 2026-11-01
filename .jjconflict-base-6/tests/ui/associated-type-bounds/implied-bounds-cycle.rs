trait A {
    type T;
}

trait B: A<T: B> {}
//~^ ERROR cycle detected when computing the implied clauses of `B`

fn main() {}
