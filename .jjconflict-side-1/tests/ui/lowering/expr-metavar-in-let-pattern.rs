//! Regression test for <https://github.com/rust-lang/rust/issues/43250>.
//! Test expr metavars aren't allowed in places where pattern is expected,
//! and their use doesn't cause ICE.

fn main() {
    let mut y;
    const C: u32 = 0;
    macro_rules! m {
        ($a:expr) => {
            let $a = 0;
        }
    }
    m!(y);
    //~^ ERROR arbitrary expressions aren't allowed in patterns
    m!(C);
    //~^ ERROR arbitrary expressions aren't allowed in patterns
}
