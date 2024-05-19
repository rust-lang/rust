trait A {
    type T;
}

trait B: A<T: B> {}
//~^ ERROR cycle detected when computing the implied predicates of `B`

fn main() {}
