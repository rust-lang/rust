type foo = { mut z : fn@() };

fn nop() { }
fn nop_foo(_x : @foo) { }

fn main() {
    let w = @{ mut z: bind nop() };
    let x = bind nop_foo(w);
    w.z = x;
}