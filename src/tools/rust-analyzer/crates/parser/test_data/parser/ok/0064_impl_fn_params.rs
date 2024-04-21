impl U {
    fn f1((a, b): (usize, usize)) {}
    fn f2(S { a, b }: S) {}
    fn f3(NewType(a): NewType) {}
    fn f4(&&a: &&usize) {}
}
