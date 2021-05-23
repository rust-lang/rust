// rustfmt-reorder_impl_items: true

impl<T> IntoIterator for SafeVec<T> {
    type Bar = u32;
    type BarFoo = u32;
    type FooBar = u32;
    // comment on FoooooBar
    type FoooooBar = u32;
    type IntoIter = self::IntoIter<T>;
    type Item = T;

    type E = impl Trait;
    type F = impl Trait;

    const AnotherConst: i32 = 100;
    const SomeConst: i32 = 100;

    // comment on foo()
    fn foo() {
        println!("hello, world");
    }

    fn foo1() {
        println!("hello, world");
    }

    fn foo2() {
        println!("hello, world");
    }

    fn foo3() {
        println!("hello, world");
    }

    fn foo4() {
        println!("hello, world");
    }

    fn foo5() {
        println!("hello, world");
    }

    fn foo6() {
        println!("hello, world");
    }

    fn foo7() {
        println!("hello, world");
    }

    fn foo8() {
        println!("hello, world");
    }
}
