// run-pass
// Test that we are able to handle the relationships between free
// regions bound in a closure callback.

#[derive(Copy, Clone)]
struct MyCx<'short, 'long: 'short> {
    short: &'short u32,
    long: &'long u32,
}

impl<'short, 'long> MyCx<'short, 'long> {
    fn short(self) -> &'short u32 { self.short }
    fn long(self) -> &'long u32 { self.long }
    fn set_short(&mut self, v: &'short u32) { self.short = v; }
}

fn with<F, R>(op: F) -> R
where
    F: for<'short, 'long> FnOnce(MyCx<'short, 'long>) -> R,
{
    op(MyCx {
        short: &22,
        long: &22,
    })
}

fn main() {
    with(|mut cx| {
        // For this to type-check, we need to be able to deduce that
        // the lifetime of `l` can be `'short`, even though it has
        // input from `'long`.
        let l = if true { cx.long() } else { cx.short() };
        cx.set_short(l);
    });
}
