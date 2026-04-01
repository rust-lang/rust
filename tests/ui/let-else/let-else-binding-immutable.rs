// from rfc2005 test suite



pub fn main() {
    let Some(x) = &Some(3) else {
        panic!();
    };
    *x += 1; //~ ERROR: cannot assign to `*x`, which is behind a `&` reference
}
