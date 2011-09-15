// error-pattern:taking the value of x will invalidate reference a

fn f(a: {mutable x: int}) -> &!int {
    ret a.x;
}

fn main() {
    let x = {mutable x: 4};
    let &a = f(x);
    x;
    a;
}
