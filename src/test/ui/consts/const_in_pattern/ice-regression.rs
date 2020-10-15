

#[derive(PartialEq)]
struct Opaque(i32);

impl Eq for Opaque {}

const FOO: &&Opaque = &&Opaque(42);

fn main() {
    match FOO {
        FOO => {}, //~ WARN must be annotated with
        // The following should not lint about unreachable
        Opaque(42) => {}
    }
}
