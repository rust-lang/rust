#![allow(incomplete_features)]
#![feature(ref_pat_everywhere)]
pub fn main() {
    if let Some(&x) = Some(0) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
    if let Some(&mut x) = Some(&0) {
        //~^ ERROR: mismatched types [E0308]
        let _: u32 = x;
    }
}
