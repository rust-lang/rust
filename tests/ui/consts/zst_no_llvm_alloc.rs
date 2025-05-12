//@ run-pass

#[repr(align(4))]
struct Foo;

static FOO: Foo = Foo;

fn main() {
    // There's no stable guarantee that these are true.
    // However, we want them to be true so that our LLVM IR and runtime are a bit faster:
    // a constant address is cheap and doesn't result in relocations in comparison to a "real"
    // global somewhere in the data section.
    let x: &'static () = &();
    assert_eq!(x as *const () as usize, 1);
    let x: &'static Foo = &Foo;
    assert_eq!(x as *const Foo as usize, 4);

    // The exact addresses returned by these library functions are not necessarily stable guarantees
    // but for now we assert that we're still matching.
    #[allow(dangling_pointers_from_temporaries)]
    {
        assert_eq!(<Vec<i32>>::new().as_ptr(), <&[i32]>::default().as_ptr());
        assert_eq!(<Box<[i32]>>::default().as_ptr(), (&[]).as_ptr());
    };

    // statics must have a unique address (see https://github.com/rust-lang/rust/issues/18297, not
    // clear whether this is a stable guarantee)
    assert_ne!(&FOO as *const Foo as usize, 4);
}
