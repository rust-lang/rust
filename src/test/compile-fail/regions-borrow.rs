fn foo(x: &uint) -> &uint { x }

fn main() {
    let p = @3u;
    let r = foo(p);
    //!^ ERROR reference is not valid
    assert *p == *r;
    //!^ ERROR reference is not valid
}
