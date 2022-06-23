#[cfg(target_pointer_width = "64")]
const N: usize = 16;

#[cfg(target_pointer_width = "32")]
const N: usize = 8;

fn main() {
    let bad = unsafe {
        std::mem::transmute::<&[u8], [u8; N]>(&[1u8])
        //~^ ERROR: type validation failed: encountered a pointer
    };
    let _val = bad[0] + bad[bad.len() - 1];
}
