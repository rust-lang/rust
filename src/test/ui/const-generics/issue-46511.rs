// check-fail

struct Foo<'a> //~ ERROR parameter `'a` is never used [E0392]
{
    _a: [u8; std::mem::size_of::<&'a mut u8>()] //~ ERROR  a non-static lifetime is not allowed in a `const`
}

pub fn main() {}
