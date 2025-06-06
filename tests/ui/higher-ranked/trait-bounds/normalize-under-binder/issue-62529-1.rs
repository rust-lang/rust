//@ check-pass

// FamilyType (GAT workaround)
pub trait FamilyLt<'a> {
    type Out;
}

struct RefMutFamily<T>(std::marker::PhantomData<T>, ());
impl<'a, T: 'a> FamilyLt<'a> for RefMutFamily<T> {
    type Out = &'a mut T;
}

pub trait Execute {
    type E: Inject;
    fn execute(self, value: <<Self::E as Inject>::I as FamilyLt>::Out);
}

pub trait Inject
where
    Self: Sized,
{
    type I: for<'a> FamilyLt<'a>;
    fn inject(_: &()) -> <Self::I as FamilyLt<'_>>::Out;
}

impl<T: 'static> Inject for RefMutFamily<T> {
    type I = Self;
    fn inject(_: &()) -> <Self::I as FamilyLt<'_>>::Out {
        unimplemented!()
    }
}

// This struct is only used to give a hint to the compiler about the type `Q`
struct Annotate<Q>(std::marker::PhantomData<Q>);
impl<Q> Annotate<Q> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

// This function annotate a closure so it can have Higher-Rank Lifetime Bounds
//
// See 'annotate' workaround: https://github.com/rust-lang/rust/issues/58052
fn annotate<F, Q>(_q: Annotate<Q>, func: F) -> impl Execute + 'static
where
    F: for<'r> FnOnce(<<Q as Inject>::I as FamilyLt<'r>>::Out) + 'static,
    Q: Inject + 'static,
{
    let wrapper: Wrapper<Q, F> = Wrapper(std::marker::PhantomData, func);
    wrapper
}

struct Wrapper<Q, F>(std::marker::PhantomData<Q>, F);
impl<Q, F> Execute for Wrapper<Q, F>
    where
        Q: Inject,
        F: for<'r> FnOnce(<<Q as Inject>::I as FamilyLt<'r>>::Out),
{
    type E = Q;

    fn execute(self, value: <<Self::E as Inject>::I as FamilyLt>::Out) {
        (self.1)(value)
    }
}

struct Task {
    _processor: Box<dyn FnOnce()>,
}

// This function consume the closure
fn task<P>(processor: P) -> Task
where P: Execute + 'static {
    Task {
        _processor: Box::new(move || {
            let q = P::E::inject(&());
            processor.execute(q);
        })
    }
}

fn main() {
    task(annotate(
        Annotate::<RefMutFamily<usize>>::new(),
        |value: &mut usize| {
            *value = 2;
        }
    ));
}
