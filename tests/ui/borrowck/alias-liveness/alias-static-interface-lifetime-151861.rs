// This test exhibits UB by creating a dangling reference from a type where the
// lifetime on a trait bound isn't live. This happens because we don't currently
// consider lifetimes in trait bounds on aliases to be live.
//
// Ultimately, the *mechanism* of creating this dangling reference is likely
// unsound, but is only exposed because of the alias-liveness change in 1.75.0.

trait Produce<'r> {
    fn produce(self) -> &'r str;
}

mod hidden {
    pub struct Wrapper(pub std::ptr::NonNull<str>);

    impl<'r> super::Produce<'r> for Wrapper {
        fn produce(self) -> &'r str {
            unsafe {
                // This is almost certainly unsound: `Wrapper` is only ever
                // built from a `&'r str` by the `construct` methods below, but
                // this impl holds at every `'r`.
                // This impl could be easily be called like:
                // let d: &'static str = {
                //   let s = String::from("..");
                //   Produce::<'static>::produce(hidden::Wrapper(NonNull::from(&*s)))
                // };
                self.0.as_ref()
            }
        }
    }
}

/// The `'static` bound is left to the user to state as a where-clause.
trait Gat {
    type Ty<'r>: Produce<'r>;
    fn construct<'r>(r: &'r str) -> Self::Ty<'r>;
}

/// The `'static` bound is an item bound on the associated type.
trait ItemBound<'r> {
    type Ty: 'static + Produce<'r>;
    fn construct(r: &'r str) -> Self::Ty;
}

struct Def;

impl Gat for Def {
    type Ty<'r> = hidden::Wrapper;
    fn construct<'r>(r: &'r str) -> Self::Ty<'r> {
        hidden::Wrapper(r.into())
    }
}

impl<'r> ItemBound<'r> for Def {
    type Ty = hidden::Wrapper;
    fn construct(r: &'r str) -> Self::Ty {
        hidden::Wrapper(r.into())
    }
}

// A generic is used in both cases so that `G::Ty` cannot be normalized.
fn item_bound<G: for<'r> ItemBound<'r>>() {
    let a;
    {
        let s = String::from("huh");
        a = <G as ItemBound>::construct(&s);
        //~^ ERROR `s` does not live long enough
    }
    let _unrelated = String::from("UB!");
    let dangling: &str = a.produce();
    println!("{dangling}");
}

fn where_clause<G: Gat>()
where
    for<'r> G::Ty<'r>: 'static,
{
    let a;
    {
        let s = String::from("huh");
        a = G::construct(&s);
        //~^ ERROR `s` does not live long enough
    }
    let _unrelated = String::from("UB!");
    let dangling: &str = a.produce();
    println!("{dangling}");
}

/// The `'static` bound is on an opaque type rather than an associated type: a method returning
/// `impl Produce<'r> + 'static`.
fn rpit_produce<'r>(r: &'r str) -> impl 'static + Produce<'r> {
    hidden::Wrapper(r.into())
}

fn rpit() {
    let a;
    {
        let s = String::from("huh");
        a = rpit_produce(&s);
        //~^ ERROR `s` does not live long enough
    }
    let _unrelated = String::from("UB!");
    let dangling: &str = a.produce();
    println!("{dangling}");
}

fn main() {
    item_bound::<Def>();
    where_clause::<Def>();
    rpit();
}
