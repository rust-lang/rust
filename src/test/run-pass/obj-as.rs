// xfail-stage1
// xfail-stage2
// xfail-stage3
obj big() {
    fn one() -> int { ret 1; }
    fn two() -> int { ret 2; }
    fn three() -> int { ret 3; }
}

type small =
    obj {
        fn one() -> int ;
    };

fn main() {
    let b: big = big();
    assert (b.one() == 1);
    assert (b.two() == 2);
    assert (b.three() == 3);
    let s: small = b as small;
    assert (s.one() == 1);
}