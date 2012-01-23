// error-pattern:fail

fn x(it: fn(int)) {
    let a = @0;
    it(1);
}

fn main() {
    x {|_x| fail; };
}