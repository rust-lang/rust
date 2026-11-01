#[allow(clippy::no_effect, clippy::unnecessary_operation)]
#[warn(clippy::int_plus_one)]
fn main() {
    let x = 1i32;
    let y = 0i32;

    let _ = x >= y + 1; //~ int_plus_one
    let _ = x >= 1 + y; //~ int_plus_one
    let _ = y + 1 <= x; //~ int_plus_one
    let _ = 1 + y <= x; //~ int_plus_one

    let _ = x - 1 >= y; //~ int_plus_one
    let _ = -1 + x >= y; //~ int_plus_one
    let _ = y <= x - 1; //~ int_plus_one
    let _ = y <= -1 + x; //~ int_plus_one

    let _ = x > y; // should be ok
    let _ = y < x; // should be ok

    // When the suggestion replaces `<=`/`>=` with `<`, an `as` cast on
    // the LHS must be parenthesized to avoid parser ambiguity
    // (e.g., `x as usize < y` is parsed as `x as usize<y>`).
    let z = 0usize;
    let _ = x as usize + 1 <= z; //~ int_plus_one
    let _ = z >= x as usize + 1; //~ int_plus_one
    // No parentheses needed when the replacement operator is `>`.
    let _ = x as usize - 1 >= z; //~ int_plus_one
    let _ = z <= x as usize - 1; //~ int_plus_one

    // Nested and parenthesized casts on the LHS.
    let _ = ((x as usize) as u8) + 1 <= 5u8; //~ int_plus_one
    let _ = (x as usize) + 1 <= z; //~ int_plus_one
}
