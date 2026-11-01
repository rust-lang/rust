//@ run-pass

struct S;
// Ensure S is moved, not copied, on assignment.
impl Drop for S { fn drop(&mut self) { } }

// user-defined function "returning" bottom (i.e., no return at all).
fn my_panic() -> ! { loop {} }

pub fn step(f: bool) {
    let mut g = S;
    let mut i = 0;
    loop
    {
        if i > 10 { break; } else { i += 1; }

        let _g = g;

        if f {
            // re-initialize g, but only before restarting loop.
            g = S;
            continue;
        }

        my_panic();

        // we never get here, so we do not need to re-initialize g.
    }
}

pub fn main() {
    step(true);
}
