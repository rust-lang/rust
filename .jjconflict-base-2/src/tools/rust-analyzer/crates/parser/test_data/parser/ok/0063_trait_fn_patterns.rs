trait T {
    fn f1((a, b): (usize, usize)) {}
    fn f2(S { a, b }: S) {}
    fn f3(NewType(a): NewType) {}
    fn f4(&&a: &&usize) {}
    fn bar(_: u64, mut x: i32);
}
