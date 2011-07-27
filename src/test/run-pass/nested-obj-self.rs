

fn main() {
    obj foo() {
        fn m1() -> str { ret "foo.m1"; }
        fn m2() -> str { ret self.m1(); }
        fn m3() -> str {
            let s1: str = self.m2();
            assert (s1 == "foo.m1");
            obj bar() {
                fn m1() -> str { ret "bar.m1"; }
                fn m2() -> str { ret self.m1(); }
            }
            let b = bar();
            let s3: str = b.m2();
            let s4: str = self.m2();
            assert (s4 == "foo.m1");
            ret s3;
        }
    }
    let a = foo();
    let s1: str = a.m1();
    assert (s1 == "foo.m1");
    let s2: str = a.m2();
    assert (s2 == "foo.m1");
    let s3: str = a.m3();
    assert (s3 == "bar.m1");
}