// issue: rust-lang/rust#99945
// ICE Failed to normalize

#![feature(type_alias_impl_trait)]

trait Widget<E> {
    type State;

    fn make_state(&self) -> Self::State;
}

impl<E> Widget<E> for () {
    type State = ();

    fn make_state(&self) -> Self::State {}
}

struct StatefulWidget<F>(F);

type StateWidget<'a> = impl Widget<&'a ()>;

impl<F: for<'a> Fn(&'a ()) -> StateWidget<'a>> Widget<()> for StatefulWidget<F> {
    type State = ();

    #[define_opaque(StateWidget)]
    fn make_state(&self) -> Self::State {}
    //~^ ERROR item does not constrain
}

#[define_opaque(StateWidget)]
fn new_stateful_widget<F: for<'a> Fn(&'a ()) -> StateWidget<'a>>(build: F) -> impl Widget<()> {
    //~^ ERROR item does not constrain
    StatefulWidget(build)
}

fn main() {
    new_stateful_widget(|_| ()).make_state();
    //~^ ERROR mismatched types
}
