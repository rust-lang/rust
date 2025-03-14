//@ revisions: legacy v0
//@[v0] compile-flags: -C symbol-mangling-version=v0
//@[legacy] compile-flags: -C symbol-mangling-version=legacy -Zunstable-options
//@ build-pass

pub struct Vec2;

pub trait Point {
    type S;
}
impl Point for Vec2 {
    type S = f32;
}

pub trait Point2: Point<S = Self::S2> {
    type S2;
}
impl Point2 for Vec2 {
    type S2 = Self::S;
}

trait MyFrom<T> {
    fn my_from();
}
impl<P: Point2> MyFrom<P::S> for P {
    fn my_from() {
        // This is just a really dumb way to force the legacy symbol mangling to
        // mangle the closure's parent impl def path *with* args. Otherwise,
        // legacy symbol mangling will strip the args from the instance, meaning
        // that we don't trigger the bug.
        let c = || {};
        let x = Box::new(c) as Box<dyn Fn()>;
    }
}

fn main() {
    <Vec2 as MyFrom<_>>::my_from();
}
