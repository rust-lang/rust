// ICE #90691 Encountered error `Unimplemented` selecting  ...
//@ build-pass
// issue: rust-lang/rust#90691

trait TError: std::fmt::Debug {}
impl TError for () {}

trait SuperTrait {
    type Error;
}

trait Trait: SuperTrait<Error: TError> {}

impl<T> Trait for T
where
    T: SuperTrait,
    <T as SuperTrait>::Error: TError,
{
}

struct SomeTrait<S>(S);
struct BoxedTrait(Box<dyn Trait<Error = ()>>);

impl<S: 'static> From<SomeTrait<S>> for BoxedTrait {
    fn from(other: SomeTrait<S>) -> Self {
        Self(Box::new(other))
    }
}

impl<S> SuperTrait for SomeTrait<S> {
    type Error = ();
}

impl From<()> for BoxedTrait {
    fn from(c: ()) -> Self {
        Self::from(SomeTrait(c))
    }
}

fn main() {
    let _: BoxedTrait = ().into();
}
