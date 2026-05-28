//@ edition: 2021

pub(crate) trait Inbox<M> {
    async fn next(self) -> M;
}

pub(crate) trait Actor: Sized {
    type Message;

    async fn on_mount(self, _: impl Inbox<Self::Message>);
}

impl<'a> Actor for () {
//~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait, self type, or predicates
    type Message = &'a ();
    async fn on_mount(self, _: impl Inbox<&'a ()>) {}
}

fn main() {}
