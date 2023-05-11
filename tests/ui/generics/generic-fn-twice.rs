// run-pass



// pretty-expanded FIXME #23616

mod foomod {
    pub fn foo<T>() { }
}

pub fn main() { foomod::foo::<isize>(); foomod::foo::<isize>(); }
