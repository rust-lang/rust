// only-aarch64
// build-fail

#![cfg_attr(bootstrap, feature(aarch64_target_feature))]

fn main() {
    #[target_feature(enable = "pacg")]
    //~^ ERROR must all be either enabled or disabled together
    unsafe fn inner() {}

    unsafe {
        foo();
        bar();
        baz();
        inner();
    }
}

#[target_feature(enable = "paca")]
//~^ ERROR must all be either enabled or disabled together
unsafe fn foo() {}


#[target_feature(enable = "paca,pacg")]
unsafe fn bar() {}

#[target_feature(enable = "paca")]
#[target_feature(enable = "pacg")]
unsafe fn baz() {}
