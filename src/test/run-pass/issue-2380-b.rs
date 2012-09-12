// xfail-fast
// aux-build:issue-2380.rs

extern mod a;

fn main() {
    a::f::<()>();
}
