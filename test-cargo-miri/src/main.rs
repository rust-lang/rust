extern crate byteorder;
#[cfg(test)]
extern crate rand;

use byteorder::{BigEndian, ByteOrder};

fn main() {
    let buf = &[1,2,3,4];
    let n = <BigEndian as ByteOrder>::read_u32(buf);
    assert_eq!(n, 0x01020304);
    println!("{:#010x}", n);
    for arg in std::env::args() {
        eprintln!("{}", arg);
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    // Make sure in-crate tests with dev-dependencies work
    #[test]
    fn rng() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xcafebeef);
        let x: u32 = rng.gen();
        let y: u32 = rng.gen();
        assert_ne!(x, y);
    }
}
