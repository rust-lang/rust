// error-pattern:invalidate reference x

fn main() {
    let v: [mut {mut x: int}] = [mut {mut x: 1}];
    for v.each {|x| v[0] = {mut x: 2}; log(debug, x); }
}
