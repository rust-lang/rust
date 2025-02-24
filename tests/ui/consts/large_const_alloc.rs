//@ only-64bit
// on 32bit and 16bit platforms it is plausible that the maximum allocation size will succeed
// FIXME (#135952) In some cases on AArch64 Linux the diagnostic does not trigger
//@ ignore-aarch64-unknown-linux-gnu

const FOO: () = {
    // 128 TiB, unlikely anyone has that much RAM
    let x = [0_u8; (1 << 47) - 1];
    //~^ ERROR evaluation of constant value failed
};

static FOO2: () = {
    let x = [0_u8; (1 << 47) - 1];
    //~^ ERROR could not evaluate static initializer
};

fn main() {
    FOO;
    FOO2;
}
