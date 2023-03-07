#[derive(Copy)]
pub struct U256(pub [u64; 4]);

impl Clone for U256 {
    fn clone(&self) -> U256 {
        *self
    }
}

impl U256 {
    pub fn new(value: u64) -> U256 {
        let mut ret = [0; 4];
        ret[0] = value;
        U256(ret)
    }
}
