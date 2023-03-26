// compile-flags: -Z unpretty=expanded
// check-pass
struct X;

fn main() {
    let x = X;
    println!("test: {x} {x:?}");
}
