//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// cc #119820, the behavior here is  inconsistent,
// using the leak check to guide inference for `for<'a> Box<_>: Leak<'a>`
// but not for `for<'a> Box<_>: IndirectLeak<'a>`.

trait Leak<'a> {}
impl Leak<'_> for Box<u32> {}
impl Leak<'static> for Box<u16> {}

fn impls_leak<T: for<'a> Leak<'a>>() {}
fn direct() {
    // ok
    //
    // The `Box<u16>` impls fails the leak check,
    // meaning that we apply the `Box<u32>` impl.
    impls_leak::<Box<_>>();
    //[next]~^ ERROR type annotations needed
}

trait IndirectLeak<'a> {}
impl<'a, T: Leak<'a>> IndirectLeak<'a> for T {}

fn impls_indirect_leak<T: for<'a> IndirectLeak<'a>>() {}
fn indirect() {
    // error: type annotations needed
    //
    // While the `Box<u16>` impl would fail the leak check
    // we have already instantiated the binder while applying
    // the generic `IndirectLeak` impl, so during candidate
    // selection of `Leak` we do not detect the placeholder error.
    // Evaluation of `Box<_>: Leak<'!a>` is therefore ambiguous,
    // resulting in `for<'a> Box<_>: Leak<'a>` also being ambiguous.
    impls_indirect_leak::<Box<_>>();
    //~^ ERROR type annotations needed
}

fn main() {}
