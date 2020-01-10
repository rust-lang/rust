// check-pass

macro_rules! def_target {
    ($target: expr) => {
        #[target_feature(enable=$target)]
        unsafe fn f() {
            #[target_feature(enable=$target)]
            ()
        }
    };
}

def_target!("avx2");

fn main() {}
