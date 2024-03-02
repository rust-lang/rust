struct Params;

pub trait Plugin<E: ?Sized> {
    type Error;
}

pub trait Pluggable {
    fn get_ref<P: Plugin<Self>>(&mut self) -> Option<P::Error> {
        None
    }
}

struct Foo;
impl Plugin<Foo> for Params {
    type Error = ();
}

impl<T: Copy> Pluggable for T {}

fn handle(req: &mut i32) {
    req.get_ref::<Params>();
    //~^ ERROR trait `Plugin<i32>` is not implemented for `Params`
}

fn main() {}
