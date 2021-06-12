// only-64bit
// on 32bit and 16bit platforms it is plausible that the maximum allocation size will succeed

const FOO: () = {
    // 128 TiB, unlikely anyone has that much RAM
    let x = [0_u8; (1 << 47) - 1];
    //~^ ERROR any use of this value will cause an error
    //~| WARNING this was previously accepted by the compiler but is being phased out
};

fn main() {
    let _ = FOO;
}
