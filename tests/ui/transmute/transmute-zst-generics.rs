// run-pass
// Transmuting to/from ZSTs that contain generics.

#![feature(transmute_generic_consts)]

unsafe fn cast_zst<T>(from: ()) -> [T; 0] {
    ::std::mem::transmute::<(), [T; 0]>(from)
}

pub fn main() {
    unsafe {
        let _: [u32; 0] = cast_zst(());
    };
}
