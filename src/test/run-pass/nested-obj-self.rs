

fn main() {
    obj foo() {
        fn m1() -> str { ret "foo.m1"; }
        fn m2() -> str { ret self.m1(); }
        fn m3() -> str {
            let str s1 = self.m2();
            assert (s1 == "foo.m1");
            obj bar() {
                fn m1() -> str { ret "bar.m1"; }
                fn m2() -> str { ret self.m1(); }
            }
            auto b = bar();
            let str s3 = b.m2();
            let str s4 = self.m2();
            assert (s4 == "foo.m1");
            ret s3;
        }
    }
    auto a = foo();
    let str s1 = a.m1();
    assert (s1 == "foo.m1");
    let str s2 = a.m2();
    assert (s2 == "foo.m1");
    let str s3 = a.m3();
    assert (s3 == "bar.m1");
}