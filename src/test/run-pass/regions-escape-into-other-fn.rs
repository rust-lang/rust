fn foo(x: &r/uint) -> &r/uint { x }
fn bar(x: &uint) -> uint { *x }

fn main() {
    let p = @3u;
    assert bar(foo(p)) == 3;
}
