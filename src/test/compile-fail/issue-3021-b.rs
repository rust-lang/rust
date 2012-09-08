use std;

fn siphash(k0 : u64) {

    struct siphash {
        mut v0: u64,
    }

    impl siphash {
        fn reset() {
           self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR attempted dynamic environment-capture
           //~^ ERROR unresolved name: k0
        }
    }
}

fn main() {}
