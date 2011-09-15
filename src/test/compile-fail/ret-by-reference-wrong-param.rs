// error-pattern:can not return a reference to the wrong parameter

fn f(a: int, b: int) -> &1 int {
    ret a;
}

fn main() {}
