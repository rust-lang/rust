//@only-target: linux # we need a specific extern supported on this target
//@normalize-stderr-test: "[48] bytes" -> "N bytes"

extern "C" {
    #[link_name = "environ"]
    static mut environ_good: i8;
    #[link_name = "environ"]
    static mut environ_bad: [i8; 10];
}

fn main() {
    let _val = unsafe { environ_good };
    let _val = unsafe { environ_bad }; //~ ERROR: /with a size of 10 bytes and alignment of 1 bytes, but Miri emulates it via an extern static shim with a size of [48] bytes and alignment of [48] bytes/
}
