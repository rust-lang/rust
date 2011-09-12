// error-pattern:fail

fn f() -> @int { fail; }

fn main() {
    let a: @int = f();
}