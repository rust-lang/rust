// The unsupported target error (E0570) does not trigger on aarch64 or arm targets
// -> so we skip this test for them
//@ ignore-aarch64
//@ ignore-arm

// gate-test-abi_c_cmse_nonsecure_call
fn main() {
    let non_secure_function = unsafe {
        core::mem::transmute::<usize, extern "C-cmse-nonsecure-call" fn(i32, i32, i32, i32) -> i32>(
            //~^ ERROR [E0658]
            //~| ERROR [E0570]
            0x10000004,
        )
    };
    let mut toto = 5;
    toto += non_secure_function(toto, 2, 3, 5);
}
