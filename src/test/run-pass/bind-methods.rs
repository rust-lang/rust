iface foo {
    fn foo() -> int;
    fn bar(p: int) -> int;
}
impl of foo for int {
    fn foo() -> int { self }
    fn bar(p: int) -> int { p * self.foo() }
}
impl <T: foo> of foo for [T] {
    fn foo() -> int { vec::foldl(0, self, {|a, b| a + b.foo()}) }
    fn bar(p: int) -> int { p + self.len() as int }
}

fn main() {
    let x = [1, 2, 3];
    let y = x.foo, z = [4, 5, 6].foo;
    assert y() + z() == 21;
    let a = x.bar, b = bind [4, 5, 6].bar(_);
    assert a(1) + b(2) + z() == 24;
}
