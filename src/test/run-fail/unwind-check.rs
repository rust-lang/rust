// error-pattern:fail

pure fn p(a: @int) -> bool { false }

fn main() {
    let a = @0;
    check p(a);
}