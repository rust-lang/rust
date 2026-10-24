// Verify that lifetime resolution correctly accounts for `Fn` bare trait objects.
//@ edition: 2015
//@ check-pass
#![allow(bare_trait_objects)]

// This should work as: fn next_u32(fill_buf: &mut dyn FnMut(&mut [u8]))
fn next_u32(fill_buf: &mut FnMut(&mut [u8])) {
    let mut buf: [u8; 4] = [0; 4];
    fill_buf(&mut buf);
}

fn explicit(fill_buf: &mut dyn FnMut(&mut [u8])) {
    let mut buf: [u8; 4] = [0; 4];
    fill_buf(&mut buf);
}

fn main() {
    let _: fn(&mut FnMut(&mut [u8])) = next_u32;
    let _: &dyn Fn(&mut FnMut(&mut [u8])) = &next_u32;
    let _: fn(&mut FnMut(&mut [u8])) = explicit;
    let _: &dyn Fn(&mut FnMut(&mut [u8])) = &explicit;
    let _: fn(&mut dyn FnMut(&mut [u8])) = next_u32;
    let _: &dyn Fn(&mut dyn FnMut(&mut [u8])) = &next_u32;
    let _: fn(&mut dyn FnMut(&mut [u8])) = explicit;
    let _: &dyn Fn(&mut dyn FnMut(&mut [u8])) = &explicit;
}
