//@only-target: linux # we need a specific extern supported on this target
//@normalize-stderr-test: "[48] bytes" -> "N bytes"

extern "C" {
    static mut environ: i8;
}

fn main() {
    let _val = unsafe { environ }; //~ ERROR: /with a size of 1 bytes and alignment of 1 bytes, but Miri emulates it via an extern static shim with a size of [48] bytes and alignment of [48] bytes/
}
