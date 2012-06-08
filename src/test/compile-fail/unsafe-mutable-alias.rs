// error-pattern:mut reference to a variable that roots another reference

fn f(a: {mut x: int}, &b: {mut x: int}) -> int {
    b.x += 1;
    ret a.x + b.x;
}

fn main() { let i = {mut x: 4}; log(debug, f(i, i)); }
