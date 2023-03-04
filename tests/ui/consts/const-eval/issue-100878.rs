// This checks that the const-eval ICE in issue #100878 does not recur.
pub fn bitshift_data(data: [u8; 1]) -> u8 {
    data[0] << 8
    //~^ ERROR: this arithmetic operation will overflow
}

fn main() {}
