// Test ensuring that `dbg!(expr)` will take ownership of the argument.

#[derive(Debug)]
struct NoCopy(usize);

fn main() {
    let a = NoCopy(0);
    let _ = dbg!(a);
    let _ = dbg!(a); //~ ERROR use of moved value
}
