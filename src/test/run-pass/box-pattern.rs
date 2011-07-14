// xfail-stage0

type foo = rec(int a, uint b);
tag bar {
    u(@foo);
    w(int);
}

fn main() {
    assert alt (u(@rec(a=10, b=40u))) {
        u(@{a, b}) { a + (b as int) }
        _ { 66 }
    } == 50;
}
