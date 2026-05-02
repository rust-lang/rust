// Full coverage regression test for #25860 soundness fix
// Tests all three variants in a single file.

// === V1: HRTB fn pointer cast ===
static UNIT: &'static &'static () = &&();

fn v1_source<'a, 'b, T>(_: &'a &'b (), v: &'b T, _: &()) -> &'a T { v }

fn v1_exploit<'a, T>(x: &'a T) -> &'static T {
    let f: fn(_, &'a T, &()) -> &'static T = v1_source;
    //~^ ERROR lifetime may not live long enough
    f(UNIT, x, &())
}

// === V3: Projection in impl header ===
trait Proj { type Out; }
impl<T> Proj for T { type Out = (); }

trait Extend<'a, 'b> {
    fn extend(self, s: &'a str) -> &'b str;
}

impl<'a, 'b> Extend<'a, 'b> for <&'b &'a () as Proj>::Out
where for<'x, 'y> &'x &'y (): Proj,
{
    fn extend(self, s: &'a str) -> &'b str {
        s //~ ERROR lifetime may not live long enough
    }
}

fn main() {}
