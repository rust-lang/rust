// error-pattern: mismatched types

struct S { a: int }
enum E { C(int) }

fn main() {
    match S { a: 1 } {
        C(_) => (),
        _ => ()
    }
}
