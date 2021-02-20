#![feature(asm, llvm_asm)]

fn main() {
    unsafe {
        asm!(".intel_syntax noprefix", "nop");
        //~^ ERROR intel syntax is the default syntax on this target
        asm!(".intel_syntax aaa noprefix", "nop");
        //~^ ERROR intel syntax is the default syntax on this target
        asm!(".att_syntax noprefix", "nop");
        //~^ ERROR using the .att_syntax directive may cause issues
        asm!(".att_syntax bbb noprefix", "nop");
        //~^ ERROR using the .att_syntax directive may cause issues
        asm!(".intel_syntax noprefix; nop");
        //~^ ERROR intel syntax is the default syntax on this target

        asm!(
            r"
            .intel_syntax noprefix
            nop"
        );
        //~^^^ ERROR intel syntax is the default syntax on this target
    }
}
