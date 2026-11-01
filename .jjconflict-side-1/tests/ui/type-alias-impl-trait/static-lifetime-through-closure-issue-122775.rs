//@ check-pass

#![feature(type_alias_impl_trait)]

fn spawn<T, F>(future: F) -> impl Sized
where
    F: FnOnce() -> T,
{
    future
}

fn spawn_task(sender: &'static ()) -> impl Sized {
    type Tait = impl Sized + 'static;
    spawn::<Tait, _>(move || sender)
}

fn main() {}
