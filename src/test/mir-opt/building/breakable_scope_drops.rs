// EMIT_MIR breakable_scope_drops.weird_temporary.built.after.mir

struct A;
struct B;

impl Drop for A {
    fn drop(&mut self) {}
}
impl Drop for B {
    fn drop(&mut self) {}
}

#[inline(always)]
fn no_unwind() {}


fn weird_temporary(a: A, b: B, nothing: ((), (), ()), x: bool) -> ((), (), ()) {
    let _temp = 'scope: {
        (
            {
                let _z = b;
                if x {
                    break 'scope nothing;
                }
            },
            match { a } {
                _ => (),
            },
            no_unwind(),
        )
    };

    nothing
}

fn main() {}
