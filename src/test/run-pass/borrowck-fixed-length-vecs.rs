// xfail-fast   (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

fn main() {
    let x = [22]/1;
    let y = &x[0];
    assert *y == 22;
}