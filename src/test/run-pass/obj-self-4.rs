


// xfail-boot
fn main() {
    obj foo(mutable int i) {
        fn inc_by(int incr) -> int { i += incr; ret i; }
        fn inc_by_5() -> int { ret self.inc_by(5); }

        // A test case showing that issue #324 is resolved.  (It used to
        // be that commenting out this (unused!) function produced a
        // type error.)
        // fn wrapper(int incr) -> int {
        //     ret self.inc_by(incr);
        // }
        fn get() -> int { ret i; }
    }
    let int rs;
    auto o = foo(5);
    rs = o.get();
    assert (rs == 5);
    rs = o.inc_by(3);
    assert (rs == 8);
    rs = o.get();
    assert (rs == 8);
}