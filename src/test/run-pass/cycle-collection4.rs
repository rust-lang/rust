type foo = { mutable z : fn@() };

fn nop() { }
fn nop_foo(_y: [int], _x : @foo) { }

fn main() {
    let w = @{ mutable z: bind nop() };
    let x = bind nop_foo([], w);
    w.z = x;
}