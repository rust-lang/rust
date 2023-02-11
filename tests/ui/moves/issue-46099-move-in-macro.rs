// Regression test for issue #46099
// Tests that we don't emit spurious
// 'value moved in previous iteration of loop' message

macro_rules! test {
    ($v:expr) => {{
        drop(&$v);
        $v
    }}
}

fn main() {
    let b = Box::new(true);
    test!({b}); //~ ERROR use of moved value
}
