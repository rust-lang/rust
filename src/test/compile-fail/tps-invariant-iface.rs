trait box_trait<T> {
    fn get() -> T;
    fn set(t: T);
}

enum box_impl<T> = {
    mut f: T
};

impl<T:copy> of box_trait<T> for box_impl<T> {
    fn get() -> T { ret self.f; }
    fn set(t: T) { self.f = t; }
}

fn set_box_trait<T>(b: box_trait<@const T>, v: @const T) {
    b.set(v);
}

fn set_box_impl<T>(b: box_impl<@const T>, v: @const T) {
    b.set(v);
}

fn main() {
    let b = box_impl::<@int>({mut f: @3});
    set_box_trait(b as box_trait::<@int>, @mut 5);
    //~^ ERROR values differ in mutability
    set_box_impl(b, @mut 5);
    //~^ ERROR values differ in mutability
}