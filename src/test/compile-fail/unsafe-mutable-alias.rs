// error-pattern:mutable reference to a variable that roots another reference

fn f(a: {mutable x: int}, &b: {mutable x: int}) -> int {
    b.x += 1;
    ret a.x + b.x;
}

fn main() { let i = {mutable x: 4}; log(debug, f(i, i)); }
