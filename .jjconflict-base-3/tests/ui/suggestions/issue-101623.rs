pub struct Stuff {
    inner: *mut (),
}

pub struct Wrap<T>(T);

fn fun<T>(t: T) -> Wrap<T> {
    todo!()
}

pub trait Trait<'de> {
    fn do_stuff(_: Wrap<&'de mut Self>);
}

impl<'a> Trait<'a> for () {
    fn do_stuff(_: Wrap<&'a mut Self>) {}
}

fn fun2(t: &mut Stuff) -> () {
    let Stuff { inner, .. } = t;
    Trait::do_stuff({ fun(&mut *inner) });
    //~^ ERROR the trait bound `*mut (): Trait<'_>` is not satisfied
}

fn main() {}
