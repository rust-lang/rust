
struct S { a: int }
enum E { C(int) }

fn main() {
    match S { a: 1 } {
        C(_) => (), //~ ERROR mismatched types: expected `S` but found `E`
        _ => ()
    }
}
