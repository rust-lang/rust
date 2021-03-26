// run-pass
trait Future: 'static {
    // The requirement for Self: Sized must prevent instantiation of
    // Future::forget in vtables, otherwise there's an infinite type
    // recursion through <Map<...> as Future>::forget.
    fn forget(self) where Self: Sized {
        Box::new(Map(self)) as Box<dyn Future>;
    }
}

struct Map<A>(A);
impl<A: Future> Future for Map<A> {}

pub struct Promise;
impl Future for Promise {}

fn main() {
    Promise.forget();
}
