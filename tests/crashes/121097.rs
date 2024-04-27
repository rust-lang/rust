//@ known-bug: #121097
#[repr(simd)]
enum Aligned {
    Zero = 0,
    One = 1,
}

fn tou8(al: Aligned) -> u8 {
    al as u8
}
