use std;

fn main() {

    obj inner() {
        fn a() -> int { ret 2; }
        fn m() -> uint { ret 3u; }
        fn z() -> uint { ret self.m(); }
    }

    let my_inner = inner();

    let my_outer =
        obj () {
            fn b() -> uint { ret 5u; }
            fn n() -> str { ret "world!"; }
            with
            my_inner
        };

    assert (my_inner.z() == 3u);
    assert (my_outer.z() == 3u);
}
