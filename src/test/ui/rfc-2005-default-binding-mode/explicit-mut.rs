// Verify the binding mode shifts - only when no `&` are auto-dereferenced is the
// final default binding mode mutable.

fn main() {
    match &&Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };

    match &mut &Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };

    match &&mut Some(5i32) {
        Some(n) => {
            *n += 1; //~ ERROR cannot assign to immutable
            let _ = n;
        }
        None => {},
    };
}
