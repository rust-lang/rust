// xfail-stage1
// error-pattern:assigning to immutable alias

fn f(&int i) {
    i += 2;
}

fn main() {
    f(1);
}