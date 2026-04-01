// This checks that the const-eval ICE in issue #100878 does not recur.
//
//@ build-pass

#[allow(arithmetic_overflow)]
pub fn bitshift_data(data: [u8; 1]) -> u8 {
    data[0] << 8
}

fn main() {}
