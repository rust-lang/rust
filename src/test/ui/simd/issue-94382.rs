#![feature(platform_intrinsics)] //~ ERROR module has missing stability attribute
#![feature(staged_api)]

struct U16x2(u16, u16);

extern "platform-intrinsic" {
    #[rustc_const_stable(feature = "foo", since = "1.3.37")]
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

fn main() {
    const U: U16x2 = U16x2(13, 14);
    const V: U16x2 = U;
    const Y0: i8 = unsafe { simd_extract(V, 0) };
}
//~^^ ERROR any use of this value will cause an error [const_err]
//~| WARN 14:29: 14:47: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
