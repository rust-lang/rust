// error-pattern:overwriting x will invalidate reference a

fn f(a: {x: {mutable x: int}}) -> &{mutable x: int} {
    ret a.x;
}

fn main() {
    let x = {x: {mutable x: 4}};
    let &a = f(x);
    x = {x: {mutable x: 5}};
    a;
}
