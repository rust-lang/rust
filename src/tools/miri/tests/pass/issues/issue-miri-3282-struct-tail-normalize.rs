// regression test for an ICE: https://github.com/rust-lang/miri/issues/3282

trait Id {
    type Assoc: ?Sized;
}

impl<T: ?Sized> Id for T {
    type Assoc = T;
}

#[repr(transparent)]
struct Foo<T: ?Sized> {
    field: <T as Id>::Assoc,
}

fn main() {
    let x = unsafe { std::mem::transmute::<fn(&str), fn(&Foo<str>)>(|_| ()) };
    let foo: &Foo<str> = unsafe { &*("uwu" as *const str as *const Foo<str>) };
    x(foo);
}
