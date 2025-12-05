#![allow(unused)]
#![warn(clippy::as_ptr_cast_mut)]
#![allow(clippy::wrong_self_convention, clippy::unnecessary_cast)]

struct MutPtrWrapper(Vec<u8>);
impl MutPtrWrapper {
    fn as_ptr(&mut self) -> *const u8 {
        self.0.as_mut_ptr() as *const u8
    }
}

struct Covariant<T>(*const T);
impl<T> Covariant<T> {
    fn as_ptr(self) -> *const T {
        self.0
    }
}

fn main() {
    let mut string = String::new();
    let _ = string.as_ptr() as *mut u8;
    //~^ as_ptr_cast_mut

    let _ = string.as_ptr() as *const i8;
    let _ = string.as_mut_ptr();
    let _ = string.as_mut_ptr() as *mut u8;
    let _ = string.as_mut_ptr() as *const u8;

    let nn = std::ptr::NonNull::new(4 as *mut u8).unwrap();
    let _ = nn.as_ptr() as *mut u8;

    let mut wrap = MutPtrWrapper(Vec::new());
    let _ = wrap.as_ptr() as *mut u8;

    let mut local = 4;
    let ref_with_write_perm = Covariant(std::ptr::addr_of_mut!(local) as *const _);
    let _ = ref_with_write_perm.as_ptr() as *mut u8;
}
