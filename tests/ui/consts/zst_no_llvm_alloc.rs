//@ run-pass

// We need some non-1 alignment to test we use the alignment of the type in the compiler.
#[repr(align(4))]
struct Foo;

static FOO: Foo = Foo;

// This tests for regression of https://github.com/rust-lang/rust/issues/147516
//
// The compiler will codegen `&Zst` without creating a real allocation, just a properly aligned
// `usize` (i.e., ptr::dangling). However, code can add an arbitrary offset from that base
// allocation. We confirm here that we correctly codegen that offset combined with the necessary
// alignment of the base &() as a 1-ZST and &Foo as a 4-ZST.
const A: *const () = (&() as *const ()).wrapping_byte_add(2);
const B: *const () = (&Foo as *const _ as *const ()).wrapping_byte_add(usize::MAX);
const C: *const () = (&Foo as *const _ as *const ()).wrapping_byte_add(2);

fn main() {
    // There's no stable guarantee that these are true.
    // However, we want them to be true so that our LLVM IR and runtime are a bit faster:
    // a constant address is cheap and doesn't result in relocations in comparison to a "real"
    // global somewhere in the data section.
    let x: &'static () = &();
    assert_eq!(x as *const () as usize, 1);
    let x: &'static Foo = &Foo;
    assert_eq!(x as *const Foo as usize, 4);

    // * A 1-aligned ZST (1-ZST) is placed at 0x1. Then offsetting that by 2 results in 3.
    // * Foo is a 4-aligned ZST, so is placed at 0x4. +2 = 6
    // * Foo is a 4-aligned ZST, so is placed at 0x4. +usize::MAX = -1 (same bit pattern) = 3
    assert_eq!(A.addr(), 3);
    assert_eq!(B.addr(), 3);
    assert_eq!(C.addr(), 6);

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
