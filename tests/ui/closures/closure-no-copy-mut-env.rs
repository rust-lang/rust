//! Checks that closures do not implement `Copy` when they capture mutable references.

fn main() {
    let mut a = 5;
    let hello = || {
        a += 1;
    };

    let b = hello;
    let c = hello; //~ ERROR use of moved value: `hello` [E0382]
}
