type foo = { mut z : fn@() };

fn nop() { }
fn nop_foo(_y: [int]/~, _x : @foo) { }

fn main() {
    let w = @{ mut z: {||nop()} };
    let x = {||nop_foo([]/~, w)};
    w.z = x;
}