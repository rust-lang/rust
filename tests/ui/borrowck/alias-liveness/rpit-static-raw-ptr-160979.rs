//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// This test exhibits UB by creating a dangling reference from a type where the
// lifetime on a trait bound isn't live. This happens because we don't currently
// consider lifetimes in trait bounds on aliases to be live.
//
// Ultimately, the *mechanism* of creating this dangling reference is likely
// unsound, but is only exposed because the lack of liveness on lifetimes in
// trait bounds on aliases.


type Payload = Box<i32>;

trait Produce<'a> {
    fn produce(self) -> &'a Payload;
}

fn make_producer<'a>(payload: &'a Payload) -> impl Produce<'a> + 'static {
    struct Producer {
        ptr: *const Payload,
    }
    impl<'a> Produce<'a> for Producer {
        fn produce(self) -> &'a Payload {
            // This is almost certainly unsound: `Wrapper` is only ever
            // built from a `&'a Payload`, but this impl holds at every `'r`.
            unsafe { &*self.ptr }
        }
    }

    Producer { ptr: payload }
}

fn main() {
    let x: Box<Payload> = Box::new(Box::new(1));
    let wrong: &'static Payload = make_producer(&x).produce();
    //~^ ERROR `x` does not live long enough
    drop(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    println!("{wrong}");
}
