//@ only-64bit
// on 32bit and 16bit platforms it is plausible that the maximum allocation size will succeed

const FOO: () = {
    // 128 TiB, unlikely anyone has that much RAM
    let x = [0_u8; (1 << 47) - 1];
    //~^ ERROR evaluation of constant value failed
};

static FOO2: () = {
    let x = [0_u8; (1 << 47) - 1];
    //^ if this errors again, remove `large_static_alloc.rs`
    // error hidden by previous error.
};

fn main() {
    FOO;
    FOO2;
}
