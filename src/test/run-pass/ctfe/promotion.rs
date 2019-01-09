// run-pass

// compile-flags: -O

fn foo(_: &'static [&'static str]) {}
fn bar(_: &'static [&'static str; 3]) {}
fn baz_i32(_: &'static i32) {}
fn baz_u32(_: &'static u32) {}

fn main() {
    foo(&["a", "b", "c"]);
    bar(&["d", "e", "f"]);

    // make sure that these do not cause trouble despite overflowing
    baz_u32(&(0-1));
    baz_i32(&-std::i32::MIN);
}
