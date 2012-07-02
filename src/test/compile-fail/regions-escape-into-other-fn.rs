fn foo(x: &uint) -> &uint { x }
fn bar(x: &uint) -> uint { *x }

fn main() {
    let p = @3u;
    bar(foo(p)); //~ ERROR reference is not valid
}
