// from rfc2005 test suite



// Verify the binding mode shifts - only when no `&` are auto-dereferenced is the
// final default binding mode mutable.

fn main() {
    let Some(n) = &&Some(5i32) else { return };
    *n += 1; //~ ERROR cannot assign to `*n`, which is behind a `&` reference
    let _ = n;

    let Some(n) = &mut &Some(5i32) else { return };
    *n += 1; //~ ERROR cannot assign to `*n`, which is behind a `&` reference
    let _ = n;

    let Some(n) = &&mut Some(5i32) else { return };
    *n += 1; //~ ERROR cannot assign to `*n`, which is behind a `&` reference
    let _ = n;
}
