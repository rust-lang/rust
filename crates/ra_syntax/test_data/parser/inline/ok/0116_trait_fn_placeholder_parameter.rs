trait Foo {
    fn bar(_: u64, mut x: i32);
    fn bar(#[attr] _: u64, #[attr] mut x: i32);
}
