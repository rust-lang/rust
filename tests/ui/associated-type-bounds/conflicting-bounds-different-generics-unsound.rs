// We previously accepted conflicting associated type bounds with different generics,
// which resulted in unsoundness.
// See https://github.com/rust-lang/rust/issues/154662

type Payload = Box<i32>;
type Src<'a> = &'a Payload;
type Dst = &'static Payload;

trait Super<T> {
    type Assoc;
}

trait Sub<'a, A1, A2>: Super<A1, Assoc = Src<'a>> + Super<A2, Assoc = Dst> {}

trait Callback<A1, A2> {
    fn callback<U: Super<A1> + Super<A2> + ?Sized>(
        payload: <U as Super<A1>>::Assoc,
    ) -> <U as Super<A2>>::Assoc;
}
struct CallbackStruct;
impl Callback<i16, i16> for CallbackStruct {
    fn callback<U: Super<i16> + ?Sized>(payload: U::Assoc) -> U::Assoc {
        payload
    }
}

fn require_trait<
    'a,
    A1,
    A2,
    U: Super<A1, Assoc = Src<'a>> + Super<A2, Assoc = Dst> + ?Sized,
    C: Callback<A1, A2>,
>(
    payload: Src<'a>,
) -> Dst {
    C::callback::<U>(payload)
}

fn use_dyn<'a, A1, A2, C: Callback<A1, A2>>(payload: Src<'a>) -> Dst {
    require_trait::<'a, A1, A2, dyn Sub<'a, A1, A2>, C>(payload)
    //^ ERROR conflicting associated type bindings for `Assoc`
}

fn extend<'a>(payload: Src<'a>) -> Dst {
    // `dyn Sub<'a, i16, i16>` has both an `Assoc = Src<'a>` bound and an `Assoc = Dst` bound.
    use_dyn::<i16, i16, CallbackStruct>(payload)
}

fn main() {
    let payload: Box<Payload> = Box::new(Box::new(1));
    let wrong: &'static Payload = extend(&*payload);
    drop(payload);
    println!("{wrong}");
}
