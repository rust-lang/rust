// Tests #108655: closures in `#[target_feature]` functions can still be marked #[inline(always)]

//@ check-pass
//@ only-x86_64

#![feature(target_feature_11)]

pub fn okay() {
    ({
        #[inline(always)]
        move || {}
    })();
}

#[target_feature(enable = "avx")]
pub unsafe fn also_okay() {
    ({
        #[inline]
        move || {}
    })();
}

#[target_feature(enable = "avx")]
pub unsafe fn warn() {
    ({
        #[inline(always)]
        //~^ WARNING: cannot use `#[inline(always)]` with `#[target_feature]` [inline_always_closure_in_target_feature_function]
        //~^^ WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        move || {}
    })();
}

#[target_feature(enable = "avx")]
pub unsafe fn also_warn() {
    ({
        move || {
            ({
                #[inline(always)]
                //~^ WARNING: cannot use `#[inline(always)]` with `#[target_feature]` [inline_always_closure_in_target_feature_function]
                //~^^ WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
                move || {}
            })();
        }
    })();
}

fn main() {}
