#![allow(unused_variables)]


fn ref_() {
    let bytes: Vec<u8> = b"Hello"[..].to_vec();
    let some = Some(bytes);

    let Some(ref a) = Some(()) else { return };

    // | ref | type annotation | & |
    // | --- | --------------- | - |
    // | x   |                 |   | OK
    // | x   |                 | x | OK
    // |     |                 | x | OK
    // | x   | x               |   |
    let Some(ref a): Option<&[u8]> = some else { return }; //~ ERROR mismatched types
    let b: & [u8] = a;

    // | x   | x               | x |
    let Some(ref a): Option<&[u8]> = &some else { return }; //~ ERROR mismatched types
    let b: & [u8] = a;

    // |     | x               |   |
    let Some(a): Option<&[u8]> = some else { return }; //~ ERROR mismatched types
    let b: &[u8] = a;
    // |     | x               | x |
    let Some(a): Option<&[u8]> = &some else { return }; //~ ERROR mismatched types
    let b: &[u8] = a;
}

fn ref_mut() {
    // This `ref mut` case had an ICE, see issue #89960
    let Some(ref mut a) = Some(()) else { return };

    let bytes: Vec<u8> = b"Hello"[..].to_vec();
    let mut some = Some(bytes);

    // | ref mut | type annotation | &mut |
    // | ------- | --------------- | ---- |
    // | x       |                 |      | OK
    // | x       |                 | x    | OK
    // |         |                 | x    | OK
    // | x       | x               |      |
    let Some(ref mut a): Option<&mut [u8]> = some else { return }; //~ ERROR mismatched types
    let b: &mut [u8] = a;

    // | x       | x               | x    | (nope)
    let Some(ref mut a): Option<&mut [u8]> = &mut some else { return }; //~ ERROR mismatched types
    let b: &mut [u8] = a;

    // |         | x               |      |
    let Some(a): Option<&mut [u8]> = some else { return }; //~ ERROR mismatched types
    let b: &mut [u8] = a;
    // |         | x               | x    |
    let Some(a): Option<&mut [u8]> = &mut some else { return }; //~ ERROR mismatched types
    let b: &mut [u8] = a;
}

fn main() {
    ref_();
    ref_mut();
}
