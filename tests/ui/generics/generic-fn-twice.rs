//@ run-pass




mod foomod {
    pub fn foo<T>() { }
}

pub fn main() { foomod::foo::<isize>(); foomod::foo::<isize>(); }
