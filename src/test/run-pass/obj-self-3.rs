

fn main() {
    obj foo() {
        fn m1(i: int) -> int { let i = i + 1; ret i; }
        fn m2(i: int) -> int { ret self.m1(i); }
        fn m3(i: int) -> int { let i = i + 1; ret self.m1(i); }
    }
    let a = foo();
    let i: int = 0;
    i = a.m1(i);
    assert (i == 1);
    i = a.m2(i);
    assert (i == 2);
    i = a.m3(i);
    assert (i == 4);
}
