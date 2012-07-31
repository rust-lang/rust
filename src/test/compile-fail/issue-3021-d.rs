use std;

trait siphash {
    fn result() -> u64;
    fn reset();
}

fn siphash(k0 : u64, k1 : u64) -> siphash {
    type sipstate = {
        mut v0 : u64,
        mut v1 : u64,
    };

    fn mk_result(st : sipstate) -> u64 {

        let v0 = st.v0,
            v1 = st.v1;
        ret v0 ^ v1;
    }

   impl of siphash for sipstate {
        fn reset() {
            self.v0 = k0 ^ 0x736f6d6570736575;  //~ ERROR attempted dynamic environment-capture
            //~^ ERROR unresolved name: k0
            self.v1 = k1 ^ 0x646f72616e646f6d;   //~ ERROR attempted dynamic environment-capture
            //~^ ERROR unresolved name: k1
        }
        fn result() -> u64 { ret mk_result(self); }
    }
}

fn main() {}
