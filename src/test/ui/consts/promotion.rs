// check-pass

// compile-flags: -O

fn foo(_: &'static [&'static str]) {}
fn bar(_: &'static [&'static str; 3]) {}
const fn baz_i32(_: &'static i32) {}
const fn baz_u32(_: &'static u32) {}

const fn fail() -> i32 { 1/0 }
const C: i32 = {
    // Promoted that fails to evaluate in dead code -- this must work
    // (for backwards compatibility reasons).
    if false {
        baz_i32(&fail());
    }
    42
};

fn main() {
    foo(&["a", "b", "c"]);
    bar(&["d", "e", "f"]);
    assert_eq!(C, 42);

    // make sure that these do not cause trouble despite overflowing
    baz_u32(&(0-1));
    baz_i32(&-i32::MIN);
}
