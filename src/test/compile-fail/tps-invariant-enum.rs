enum box_impl<T> = {
    mut f: T
};

fn set_box_impl<T>(b: box_impl<@const T>, v: @const T) {
    b.f = v;
}

fn main() {
    let b = box_impl::<@int>({mut f: @3});
    set_box_impl(b, @mut 5);
    //~^ ERROR values differ in mutability

    // No error when type of parameter actually IS @const int
    let x: @const int = @3; // only way I could find to upcast
    let b = box_impl::<@const int>({mut f: x});
    set_box_impl(b, @mut 5);
}