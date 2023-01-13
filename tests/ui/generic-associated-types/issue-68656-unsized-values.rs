// Regression test for #68656

trait UnsafeCopy<T: Copy> {
    type Item<'a>: std::ops::Deref<Target = T>;

    fn bug<'a>(item: &Self::Item<'a>) -> () {
        let x: T = **item;
        &x as *const _;
    }
}

impl<T: Copy + std::ops::Deref> UnsafeCopy<T> for T {
    type Item<'a> = T;
    //~^ ERROR type mismatch resolving `<T as Deref>::Target == T`
}

fn main() {
    <&'static str>::bug(&"");
}
