// error-pattern:fail

fn f(a: @int) {
    fail;
}

fn main() {
    let b = @0;
    let g = {|move b|f(b)};
    g();
}