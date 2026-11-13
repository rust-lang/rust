//@ run-pass

static mut TEST: i32 = {
    // We must not promote this, as CTFE needs to be able to mutate it later.
    let x = &mut [1,2,3];
    x[0] += 1;
    x[0]
};

// This still works -- it's not done via promotion.
#[allow(unused)]
static mut TEST2: &'static mut [i32] = &mut [0,1,2];

fn main() {
    assert_eq!(unsafe { TEST }, 2);
}
