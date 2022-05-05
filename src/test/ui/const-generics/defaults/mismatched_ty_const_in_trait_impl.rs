trait Trait {
    fn foo<U>() {}
}
impl Trait for () {
    fn foo<const M: u64>() {}
    //~^ error: method `foo` has an incompatble generic parameter for trait
}

trait Other {
    fn bar<const M: u8>() {}
}
impl Other for () {
    fn bar<T>() {}
    //~^ error: method `bar` has an incompatible generic parameter for trait
}

trait Uwu {
    fn baz<const N: u32>() {}
}
impl Uwu for () {
    fn baz<const N: i32>() {}
    //~^ error: method `baz` has an incompatible generic parameter for trait
}

fn main() {}
