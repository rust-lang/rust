

fn main() {
    obj foo() {
        fn m1(int i) -> int { i += 1; ret i; }
        fn m2(int i) -> int { ret self.m1(i); }
        fn m3(int i) -> int { i += 1; ret self.m1(i); }
    }
    auto a = foo();
    let int i = 0;
    i = a.m1(i);
    assert (i == 1);
    i = a.m2(i);
    assert (i == 2);
    i = a.m3(i);
    assert (i == 4);
}