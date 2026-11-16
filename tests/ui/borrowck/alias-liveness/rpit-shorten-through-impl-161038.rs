//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next
//@ [polonius] check-pass
//@ [polonius] known-bug: #161038

// Two different variations of #161038: one with a trait with a method taking
// a reference and one taking an impl Trait. Fundamentally, both exploit the
// fact that we don't treat `'a` as live in `Produce<'a>` bound in the return
// type.

#![forbid(unsafe_code)]

type Payload = Box<i32>;

trait Produce<'a> {
    fn produce(self) -> &'a Payload;
}
impl<'a> Produce<'a> for &'a Payload {
    fn produce(self) -> &'a Payload {
        self
    }
}

trait OutlivesAlias<'a, 'b> {
    fn shorten(x: impl Produce<'a> + 'a) -> impl Produce<'a> + 'b;
}
impl<'a: 'b, 'b> OutlivesAlias<'a, 'b> for () {
    fn shorten(x: impl Produce<'a> + 'a) -> impl Produce<'a> + 'b {
        x
    }
}

fn make_producer_alias<'a, 'b>(payload: &'a Payload) -> impl Produce<'a> + 'b + use<'a, 'b>
where
    (): OutlivesAlias<'a, 'b>,
{
    <() as OutlivesAlias<'a, 'b>>::shorten(payload)
}

trait OutlivesRef<'a, 'b> {
    fn shorten(x: &'a Payload) -> impl Produce<'a> + 'b;
}
impl<'a: 'b, 'b> OutlivesRef<'a, 'b> for () {
    fn shorten(x: &'a Payload) -> impl Produce<'a> + 'b {
        x
    }
}

fn make_producer_ref<'a, 'b>(payload: &'a Payload) -> impl Produce<'a> + 'b + use<'a, 'b>
where
    (): OutlivesRef<'a, 'b>,
{
    <() as OutlivesRef<'a, 'b>>::shorten(payload)
}

fn through_alias_param() {
    let wrong: &Payload;
    {
        let x: Box<Payload> = Box::new(Box::new(1));
        let r: &Payload = &*x; //[nll]~ ERROR `*x` does not live long enough
        wrong = make_producer_alias(r).produce();
    }
    println!("{wrong}");
}

fn through_impl_header() {
    let wrong: &Payload;
    {
        let x: Box<Payload> = Box::new(Box::new(1));
        let r: &Payload = &*x; //[nll]~ ERROR `*x` does not live long enough
        wrong = make_producer_ref(r).produce();
    }
    println!("{wrong}");
}

fn main() {
    through_alias_param();
    through_impl_header();
}
