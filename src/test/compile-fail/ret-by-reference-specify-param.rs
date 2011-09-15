// error-pattern:must specify referenced parameter

fn f(a: int, b: int) -> &int {
    ret a;
}

fn main() {}
