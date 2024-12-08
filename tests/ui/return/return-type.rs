struct S<T> {
    t: T,
}

fn foo<T>(x: T) -> S<T> {
    S { t: x }
}

fn bar() {
    foo(4 as usize)
    //~^ ERROR mismatched types
}

fn main() {}
