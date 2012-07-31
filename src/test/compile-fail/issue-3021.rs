use std;

trait siphash {
    fn reset();
}

fn siphash(k0 : u64) -> siphash {
    type sipstate = {
        mut v0 : u64,
    };


   impl of siphash for sipstate {
        fn reset() {
           self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR attempted dynamic environment-capture
           //~^ ERROR unresolved name: k0
        }
    }
    fail;
}

fn main() {}
