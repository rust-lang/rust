trait Trait {
    fn foo<U>() {}
}
impl Trait for () {
    fn foo<const M: u64>() {}
    //~^ error: associated function `foo` has an incompatible generic parameter for trait
}

trait Other {
    fn bar<const M: u8>() {}
}
impl Other for () {
    fn bar<T>() {}
    //~^ error: associated function `bar` has an incompatible generic parameter for trait
}

trait Uwu {
    fn baz<const N: u32>() {}
}
impl Uwu for () {
    fn baz<const N: i32>() {}
    //~^ error: associated function `baz` has an incompatible generic parameter for trait
}

trait Aaaaaa {
    fn bbbb<const N: u32, T>() {}
}
impl Aaaaaa for () {
    fn bbbb<T, const N: u32>() {}
    //~^ error: associated function `bbbb` has an incompatible generic parameter for trait
}

trait Names {
    fn abcd<T, const N: u32>() {}
}
impl Names for () {
    fn abcd<const N: u32, T>() {}
    //~^ error: associated function `abcd` has an incompatible generic parameter for trait
}

fn main() {}
