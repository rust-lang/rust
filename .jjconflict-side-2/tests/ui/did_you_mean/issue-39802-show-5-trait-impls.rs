trait Foo<B> {
    fn bar(&self){}
}

impl Foo<u8> for i8 {}
impl Foo<u16> for i8 {}
impl Foo<u32> for i8 {}
impl Foo<u64> for i8 {}
impl Foo<bool> for i8 {}

impl Foo<u16> for u8 {}
impl Foo<u32> for u8 {}
impl Foo<u64> for u8 {}
impl Foo<bool> for u8 {}

impl Foo<u8> for bool {}
impl Foo<u16> for bool {}
impl Foo<u32> for bool {}
impl Foo<u64> for bool {}
impl Foo<bool> for bool {}
impl Foo<i8> for bool {}

fn main() {
    Foo::<i32>::bar(&1i8); //~ ERROR is not satisfied
    Foo::<i32>::bar(&1u8); //~ ERROR is not satisfied
    Foo::<i32>::bar(&true); //~ ERROR is not satisfied
}
