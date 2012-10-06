// compile-flags:-Z no-opt
use comm::*;

// This test has to be setup just so to trigger
// the condition which was causing us a crash.
// The situation is that we are capturing a
// () value by ref.  We generally feel free,
// however, to substitute NULL pointers and
// undefined values for values of () type, and
// so this caused a segfault when we copied into
// the closure.
//
// The fix is just to not emit any actual loads
// or stores for copies of () type (which is of
// course preferable, as the value itself is
// irrelevant).

fn foo(&&x: ()) -> Port<()> {
    let p = Port();
    let c = Chan(&p);
    do task::spawn() |copy c, copy x| {
        c.send(x);
    }
    p
}

fn main() {
    foo(()).recv()
}
