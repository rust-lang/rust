//@ check-pass



fn main() {
    let Some(n) = &mut &mut Some(5i32) else { return; };
    *n += 1; // OK
    let _ = n;

    let Some(n): &mut Option<i32> = &mut &mut Some(5i32) else { return; };
    *n += 1; // OK
    let _ = n;
}
