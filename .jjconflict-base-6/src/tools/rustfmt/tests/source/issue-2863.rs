// rustfmt-reorder_impl_items: true

impl<T> IntoIterator for SafeVec<T> {
    type F = impl Trait;
    type IntoIter = self::IntoIter<T>;
    type Item = T;
    // comment on foo()
    fn foo() {println!("hello, world");}
    type Bar = u32;
    fn foo1() {println!("hello, world");}
    type FooBar = u32;
    fn foo2() {println!("hello, world");}
    fn foo3() {println!("hello, world");}
    const SomeConst: i32 = 100;
    fn foo4() {println!("hello, world");}
    fn foo5() {println!("hello, world");}
    // comment on FoooooBar
    type FoooooBar = u32;
    fn foo6() {println!("hello, world");}
    fn foo7() {println!("hello, world");}
    type BarFoo = u32;
    type E = impl Trait;
    const AnotherConst: i32 = 100;
    fn foo8() {println!("hello, world");}
}
