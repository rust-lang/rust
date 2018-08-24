fn siphash(k0 : u64) {

    struct siphash {
        v0: u64,
    }

    impl siphash {
        pub fn reset(&mut self) {
           self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR can't capture dynamic environment
        }
    }
}

fn main() {}
