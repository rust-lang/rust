//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

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
            unsafe { &*self.ptr }
        }
    }

    // SAFETY: a user of this can only produce a `&'a Payload` from it... right?
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
