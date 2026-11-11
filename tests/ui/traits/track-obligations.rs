// These are simplifications of the tower traits by the same name:

pub trait Service<Request> {
    type Response;
}

pub trait Layer<C> {
    type Service;
}

// Any type will do here:

pub struct Req;
pub struct Res;

// This is encoding a trait alias.

pub trait ParticularService:
    Service<Req, Response = Res> {
}

impl<T> ParticularService for T
where
    T: Service<Req, Response = Res>,
{
}

// This is also a trait alias.
// The weird = <Self as ...> bound is there so that users of the trait do not
// need to repeat the bounds. See https://github.com/rust-lang/rust/issues/20671
// for context, and in particular the workaround in:
// https://github.com/rust-lang/rust/issues/20671#issuecomment-529752828

pub trait ParticularServiceLayer<C>:
    Layer<C, Service = <Self as ParticularServiceLayer<C>>::Service>
{
    type Service: ParticularService;
}

impl<T, C> ParticularServiceLayer<C> for T
where
    T: Layer<C>,
    T::Service: ParticularService,
{
    type Service = T::Service;
}

// These are types that implement the traits that the trait aliases refer to.
// They should also implement the alias traits due to the blanket impls.

struct ALayer<C>(C);
impl<C> Layer<C> for ALayer<C> {
    type Service = AService;
}

struct AService;
impl Service<Req> for AService {
    // However, AService does _not_ meet the blanket implementation,
    // since its Response type is bool, not Res as it should be.
    type Response = bool;
}

// This is a wrapper type around ALayer that uses the trait alias
// as a way to communicate the requirements of the provided types.
struct Client<C>(C);

// The method and the free-standing function below both have the same bounds.

impl<C> Client<C>
where
    ALayer<C>: ParticularServiceLayer<C>,
{
    fn check(&self) {}
}

fn check<C>(_: C) where ALayer<C>: ParticularServiceLayer<C> {}

// But, they give very different error messages.

fn main() {
    // This gives a very poor error message that does nothing to point the user
    // at the underlying cause of why the types involved do not meet the bounds.
    Client(()).check(); //~ ERROR E0599

    // This gives a good(ish) error message that points the user at _why_ the
    // bound isn't met, and thus how they might fix it.
    check(()); //~ ERROR E0271
}
