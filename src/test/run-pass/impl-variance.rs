impl extensions<T> for [const T] {
    fn foo() -> uint { vec::len(self) }
}

fn main() {
    let v = [const 0];
    assert v.foo() == 1u;
    let v = [0];
    assert v.foo() == 1u;
    let v = [mut 0];
    assert v.foo() == 1u;
}