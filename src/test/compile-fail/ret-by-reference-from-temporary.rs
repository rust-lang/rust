// error-pattern:a reference binding can't be rooted in a temporary

fn f(a: {x: int}) -> &int {
    ret a.x;
}

fn main() {
    let &_a = f({x: 4});
}
