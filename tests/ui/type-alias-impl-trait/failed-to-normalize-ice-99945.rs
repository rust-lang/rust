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

    fn make_state(&self) -> Self::State {}
}

fn new_stateful_widget<F: for<'a> Fn(&'a ()) -> StateWidget<'a>>(build: F) -> impl Widget<()> {
    StatefulWidget(build)
    //~^ ERROR expected generic lifetime parameter, found `'a`
}

fn main() {
    new_stateful_widget(|_| ()).make_state();
    //~^ ERROR mismatched types
}
