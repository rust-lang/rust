//@ run-pass

#[repr(align(4))]
struct Foo;

static FOO: Foo = Foo;

fn main() {
    let x: &'static () = &();
    assert_ne!(x as *const () as usize, 1);
    let x: &'static Foo = &Foo;
    assert_ne!(x as *const Foo as usize, 4);

    // statics must have a unique address
    assert_ne!(&FOO as *const Foo as usize, 4);

    // FIXME this two tests should be assert_eq!
    // this stopped working since we are promoting to constants instead of statics
    assert_ne!(<Vec<i32>>::new().as_ptr(), <&[i32]>::default().as_ptr());
    assert_ne!(<Box<[i32]>>::default().as_ptr(), (&[]).as_ptr());
}
