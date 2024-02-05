// only-64bit
// on 32bit and 16bit platforms it is plausible that the maximum allocation size will succeed

static FOO2: () = {
    let x = [0_u8; (1 << 47) - 1];
    //~^ ERROR could not evaluate static initializer
};

fn main() {
    FOO2;
}
