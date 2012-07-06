fn foo(x: &uint) -> &uint { x }
fn bar(x: &uint) -> uint { *x }

fn main() {
    let p = @3u;
    assert bar(foo(p)) == 3;
}
