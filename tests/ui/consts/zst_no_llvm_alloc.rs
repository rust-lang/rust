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
    assert_eq!(x as *const () as isize, isize::MIN);
    let x: &'static Foo = &Foo;
    assert_eq!(x as *const Foo as isize, isize::MIN);
    // We also put ZST locals at the same fake address, rather than `alloca`ing
    let local_unit = ();
    assert_eq!((&raw const local_unit) as isize, isize::MIN);
    let local_high_align_zst = [0_u128; 0];
    assert_eq!((&raw const local_high_align_zst) as isize, isize::MIN);

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
