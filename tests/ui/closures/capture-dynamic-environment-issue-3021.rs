// https://github.com/rust-lang/rust/issues/3021
// Tests that closures cannot capture a variable from the dynamic environment in this context.
fn siphash(k0 : u64) {

    struct SipHash {
        v0: u64,
    }

    impl SipHash {
        pub fn reset(&mut self) {
           self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR can't capture dynamic environment
        }
    }
}

fn main() {}
