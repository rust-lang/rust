//@check-pass
#![warn(clippy::cast_lossless)]

fn issue15348() {
    macro_rules! zero {
        ($int:ty) => {{
            let data: [u8; 3] = [0, 0, 0];
            data[0] as $int
        }};
    }

    let _ = zero!(u8);
    let _ = zero!(u16);
    let _ = zero!(u32);
    let _ = zero!(u64);
    let _ = zero!(u128);
}
