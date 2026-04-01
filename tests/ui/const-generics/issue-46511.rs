//@ check-fail

struct Foo<'a> //~ ERROR parameter `'a` is never used [E0392]
{
    _a: [u8; std::mem::size_of::<&'a mut u8>()] //~ ERROR generic parameters may not be used in const operations
}

pub fn main() {}
