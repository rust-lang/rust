// needs-asm-support
// only-x86_64

// checks various modes of failure for the `clobber_abi` argument (after parsing)

#![feature(asm)]

fn main() {
    unsafe {
        asm!("", clobber_abi("C"));
        asm!("", clobber_abi("foo"));
        //~^ ERROR invalid ABI for `clobber_abi`
        asm!("", clobber_abi("C", "foo"));
        //~^ ERROR invalid ABI for `clobber_abi`
        asm!("", clobber_abi("C", "C"));
        //~^ ERROR `C` ABI specified multiple times
        asm!("", clobber_abi("win64", "sysv64"));
        asm!("", clobber_abi("win64", "efiapi"));
        //~^ ERROR `win64` ABI specified multiple times
        asm!("", clobber_abi("C", "foo", "C"));
        //~^ ERROR invalid ABI for `clobber_abi`
        //~| ERROR `C` ABI specified multiple times
        asm!("", clobber_abi("win64", "foo", "efiapi"));
        //~^ ERROR invalid ABI for `clobber_abi`
        //~| ERROR `win64` ABI specified multiple times
        asm!("", clobber_abi("C"), clobber_abi("C"));
        //~^ ERROR `C` ABI specified multiple times
        asm!("", clobber_abi("win64"), clobber_abi("sysv64"));
        asm!("", clobber_abi("win64"), clobber_abi("efiapi"));
        //~^ ERROR `win64` ABI specified multiple times
    }
}
