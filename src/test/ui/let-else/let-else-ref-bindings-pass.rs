// check-pass

#![feature(let_else)]
#![allow(unused_variables)]

fn ref_() {
    let bytes: Vec<u8> = b"Hello"[..].to_vec();
    let some = Some(bytes);

    let Some(ref a) = Some(()) else { return };

    // | ref | type annotation | & |
    // | --- | --------------- | - |
    // | x   | x               |   | error
    // | x   | x               | x | error
    // |     | x               |   | error
    // |     | x               | x | error
    // | x   |                 |   |
    let Some(ref a) = some else { return }; // OK
    let b: &[u8] = a;

    // | x   |                 | x |
    let Some(ref a) = &some else { return }; // OK
    let b: &[u8] = a;


    // |     |                 | x |
    let Some(a) = &some else { return }; // OK
    let b: &[u8] = a;

    let Some(a): Option<&[u8]> = some.as_deref() else { return }; // OK
    let b: &[u8] = a;
    let Some(ref  a): Option<&[u8]> = some.as_deref() else { return }; // OK
    let b: &[u8] = a;
}

fn ref_mut() {
    // This `ref mut` case had an ICE, see issue #89960
    let Some(ref mut a) = Some(()) else { return };

    let bytes: Vec<u8> = b"Hello"[..].to_vec();
    let mut some = Some(bytes);

    // | ref mut | type annotation | &mut |
    // | ------- | --------------- | ---- |
    // | x       | x               |      | error
    // | x       | x               | x    | error
    // |         | x               |      | error
    // |         | x               | x    | error
    // | x       |                 |      |
    let Some(ref mut a) = some else { return }; // OK
    let b: &mut [u8] = a;

    // | x       |                 | x    |
    let Some(ref mut a) = &mut some else { return }; // OK
    let b: &mut [u8] = a;

    // |         |                 | x    |
    let Some(a) = &mut some else { return }; // OK
    let b: &mut [u8] = a;

    let Some(a): Option<&mut [u8]> = some.as_deref_mut() else { return }; // OK
    let b: &mut [u8] = a;
    let Some(ref mut a): Option<&mut [u8]> = some.as_deref_mut() else { return }; // OK
    let b: &mut [u8] = a;
}

fn main() {
    ref_();
    ref_mut();
}
