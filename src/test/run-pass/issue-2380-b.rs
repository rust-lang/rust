// xfail-fast
// aux-build:issue-2380.rs

use a;

fn main() {
    a::f::<()>();
}
