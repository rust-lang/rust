//@ check-pass

// Repeated calls to the same monomorphic `const fn` must keep using the same
// local layouts. This would fail at compile time if the instance cache mixed
// layouts across instances (e.g. `ident::<u64>` vs `ident::<bool>`).

const fn bump(x: u64) -> u64 {
    x.wrapping_add(1)
}

const fn ident<T: Copy>(x: T) -> T {
    x
}

const fn bits(v: u64) -> f64 {
    f64::from_bits(v)
}

const TABLE: [u64; 32] = {
    let mut t = [0u64; 32];
    let mut i = 0;
    while i < 32 {
        t[i] = bump(ident(i as u64));
        i += 1;
    }
    t
};

const FLOATS: [f64; 8] = {
    let mut t = [0.0; 8];
    let mut i = 0;
    while i < 8 {
        t[i] = bits(i as u64);
        i += 1;
    }
    t
};

const _: () = assert!(TABLE[0] == 1);
const _: () = assert!(TABLE[31] == 32);
const _: () = assert!(FLOATS[0].to_bits() == 0);
const _: () = assert!(ident(true) == true);
const _: () = assert!(ident(3u8) == 3);

fn main() {}
