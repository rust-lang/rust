// error-pattern:mutable alias to a variable that roots another alias

fn f(a: {mutable x: int}, &b: {mutable x: int}) -> int {
    b.x += 1;
    ret a.x + b.x;
}

fn main() { let i = {mutable x: 4}; log f(i, i); }
