


// -*- rust -*-
mod foomod {
    #[legacy_exports];
    fn foo<T>() { }
}

fn main() { foomod::foo::<int>(); foomod::foo::<int>(); }
