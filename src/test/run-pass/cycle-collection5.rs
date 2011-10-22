type foo = { mutable z : fn@() };

fn nop() { }
fn nop_foo(_y: o, _x : @foo) { }

obj o() {
}

fn main() {
    let w = @{ mutable z: bind nop() };
    let x = bind nop_foo(o(), w);
    w.z = x;
}