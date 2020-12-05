trait SipHash {
    fn result(&self) -> u64;
    fn reset(&self);
}

fn siphash(k0 : u64, k1 : u64) {
    struct SipState {
        v0: u64,
        v1: u64,
    }

    fn mk_result(st : &SipState) -> u64 {

        let v0 = st.v0;
        let v1 = st.v1;
        return v0 ^ v1;
    }

   impl SipHash for SipState {
        fn reset(&self) {
            self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR can't capture dynamic environment
            self.v1 = k1 ^ 0x646f72616e646f6d; //~ ERROR can't capture dynamic environment
        }
        fn result(&self) -> u64 { return mk_result(self); }
    }
}

fn main() {}
