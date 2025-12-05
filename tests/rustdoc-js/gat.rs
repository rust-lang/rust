pub trait Foo {
    type Assoc<T>;
}

pub fn sample<X: Foo<Assoc<u8> = u8>>(_: X) -> u32 {
    loop {}
}
pub fn synergy(_: impl Foo<Assoc<u8> = u8>) -> ! {
    loop {}
}
pub fn consider(_: impl Foo<Assoc<u8> = u32>) -> bool {
    loop {}
}
pub fn integrate<T>(_: impl Foo<Assoc<T> = T>) -> T {
    loop {}
}
