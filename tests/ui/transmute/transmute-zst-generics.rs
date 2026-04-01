//@ run-pass

// Transmuting to/from ZSTs that contain generics.

#![feature(transmute_generic_consts)]

// Verify non-generic ZST -> generic ZST transmute
unsafe fn cast_zst0<T>(from: ()) -> [T; 0] {
    ::std::mem::transmute::<(), [T; 0]>(from)
}

// Verify generic ZST -> non-generic ZST transmute
unsafe fn cast_zst1<T>(from: [T; 0]) -> () {
    ::std::mem::transmute::<[T; 0], ()>(from)
}

// Verify transmute with generic compound types
unsafe fn cast_zst2<T>(from: ()) -> [(T, T); 0] {
    ::std::mem::transmute::<(), [(T, T); 0]>(from)
}

// Verify transmute with ZST propagation through arrays
unsafe fn cast_zst3<T>(from: ()) -> [[T; 0]; 8] {
    ::std::mem::transmute::<(), [[T; 0]; 8]>(from)
}

// Verify transmute with an extra ZST field
pub struct PtrAndZst<T: ?Sized> {
    _inner: *mut T,
    _other: (),
}
pub unsafe fn cast_ptr<T: ?Sized>(from: *mut T) -> PtrAndZst<T> {
    std::mem::transmute(from)
}

pub fn main() {
    unsafe {
        let _: [u32; 0] = cast_zst0(());
        let _ = cast_zst1::<u32>([]);
        let _: [(u32, u32); 0] = cast_zst2(());
        let _: [[u32; 0]; 8] = cast_zst3(());
        cast_ptr(&mut 42);
    };
}
