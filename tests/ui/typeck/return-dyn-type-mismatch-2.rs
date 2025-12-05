trait Trait<T> {}

fn foo<T>() -> dyn Trait<T>
where
    dyn Trait<T>: Sized, // pesky sized predicate
{
    42
    //~^ ERROR mismatched types
}

fn main() {}
