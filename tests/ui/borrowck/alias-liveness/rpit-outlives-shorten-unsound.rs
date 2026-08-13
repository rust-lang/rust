//@ revisions: nll polonius
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next
//@ edition: 2024

// Regression test for #161038: Polonius accepted an illegal lifetime
// shortening through `impl Trait`. `Outlives::shorten` turns
// `impl Produce<'a> + 'a` into `impl Produce<'a> + 'b` (`'a: 'b` on the
// impl). The `'b` outlives bound made liveness treat `'a` as dead even
// though `.produce()` still yields `&'a Payload`, so a reference to
// dropped data was returned. NLL already rejected this; Polonius must too.

type Payload = Box<i32>;

trait Produce<'a> {
    fn produce(self) -> &'a Payload;
}
impl<'a> Produce<'a> for &'a Payload {
    fn produce(self) -> &'a Payload {
        self
    }
}

trait Outlives<'a, 'b> {
    fn shorten(x: impl Produce<'a> + 'a) -> impl Produce<'a> + 'b;
}
impl<'a: 'b, 'b> Outlives<'a, 'b> for () {
    fn shorten(x: impl Produce<'a> + 'a) -> impl Produce<'a> + 'b {
        x
    }
}

fn make_producer<'a, 'b>(payload: &'a Payload) -> impl Produce<'a> + 'b + use<'a, 'b>
where
    (): Outlives<'a, 'b>,
{
    <()>::shorten(payload)
}

fn main() {
    let wrong: &Payload;
    {
        let x: Box<Payload> = Box::new(Box::new(1));
        let r: &Payload = &*x;
        //~^ ERROR `*x` does not live long enough
        wrong = make_producer(r).produce();
    }
    let _ = wrong;
}
