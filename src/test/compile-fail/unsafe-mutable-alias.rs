// xfail-stage0
// error-pattern:mutable alias to a variable that roots another alias

fn f(&int a, &mutable int b) -> int {
    b += 1;
    ret a + b;
}

fn main() {
    auto i = 4;
    log f(i, i);
}
