fn main() {
    alt {a: 10, b: @20} {
        x@{a, b: @20} { assert x.a == 10; assert a == 10; }
        {b, _} { fail; }
    }
    let x@{b, _} = {a: 10, b: {mutable c: 20}};
    x.b.c = 30;
    assert b.c == 20;
    let y@{d, _} = {a: 10, d: {mutable c: 20}};
    y.d.c = 30;
    assert d.c == 20;
}
