class box_impl<T> {
    let mut f: T;

    new(f: T) {
        self.f = f;
    }
}

fn set_box_impl<T>(b: box_impl<@const T>, v: @const T) {
    b.f = v;
}

fn main() {
    let b = box_impl::<@int>(@3);
    set_box_impl(b, @mut 5);
    //~^ ERROR values differ in mutability

    // No error when type of parameter actually IS @const int
    let b = box_impl::<@const int>(@3);
    set_box_impl(b, @mut 5);
}