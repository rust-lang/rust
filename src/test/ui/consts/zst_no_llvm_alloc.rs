// run-pass

#[repr(align(4))]
struct Foo;

static FOO: Foo = Foo;

fn main() {
    let x: &'static () = &();
    assert_eq!(x as *const () as usize, 1);
    let x: &'static Foo = &Foo;
    assert_eq!(x as *const Foo as usize, 4);

    // statics must have a unique address
    assert_ne!(&FOO as *const Foo as usize, 4);

    assert_eq!(<Vec<i32>>::new().as_ptr(), <&[i32]>::default().as_ptr());
    assert_eq!(<Box<[i32]>>::default().as_ptr(), (&[]).as_ptr());
}
