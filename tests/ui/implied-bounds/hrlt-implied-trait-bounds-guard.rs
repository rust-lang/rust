// A test exploiting the bug behind #25860 except with
// implied trait bounds which currently don't exist.
use std::marker::PhantomData;
struct Foo<'a, 'b, T>(PhantomData<(&'a (), &'b (), T)>)
where
    T: Convert<'a, 'b>;

trait Convert<'a, 'b>: Sized {
    fn cast(&'a self) -> &'b Self;
}
impl<'long: 'short, 'short, T> Convert<'long, 'short> for T {
    fn cast(&'long self) -> &'short T {
        self
    }
}

// This function will compile once we add implied trait bounds.
//
// If we're not careful with our impl, the transformations
// in `bad` would succeed, which is unsound ✨
//
// FIXME: the error is pretty bad, this should say
//
//     `T: Convert<'in_, 'out>` is not implemented
//
// help: needed by `Foo<'in_, 'out, T>`
//
// Please ping @lcnr if your changes end up causing `badboi` to compile.
fn badboi<'in_, 'out, T>(x: Foo<'in_, 'out, T>, sadness: &'in_ T) -> &'out T {
    //~^ ERROR lifetime mismatch
    sadness.cast()
    //~^ ERROR may not live long enough
}

fn badboi2<'in_, 'out, T>(x: Foo<'in_, 'out, T>, sadness: &'in_ T) {
    //~^ ERROR lifetime mismatch
    let _: &'out T = sadness.cast();
    //~^ ERROR may not live long enough
}

fn badboi3<'in_, 'out, T>(a: Foo<'in_, 'out, (&'in_ T, &'out T)>, sadness: &'in_ T) {
    //~^ ERROR lifetime mismatch
    let _: &'out T = sadness.cast();
    //~^ ERROR may not live long enough
}

fn bad<'short, T>(value: &'short T) -> &'static T {
    let x: for<'in_, 'out> fn(Foo<'in_, 'out, T>, &'in_ T) -> &'out T = badboi;
    let x: for<'out> fn(Foo<'short, 'out, T>, &'short T) -> &'out T = x;
    let x: for<'out> fn(Foo<'static, 'out, T>, &'short T) -> &'out T = x;
    let x: fn(Foo<'static, 'static, T>, &'short T) -> &'static T = x;
    x(Foo(PhantomData), value)
}

// Use `bad` to cause a segfault.
fn main() {
    let mut outer: Option<&'static u32> = Some(&3);
    let static_ref: &'static &'static u32 = match outer {
        Some(ref reference) => bad(reference),
        None => unreachable!(),
    };
    outer = None;
    println!("{}", static_ref);
}
