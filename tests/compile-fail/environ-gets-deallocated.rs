extern "C" {
    static environ: *const *const u8;
}

fn main() {
    let pointer = unsafe { environ };
    let _x = unsafe { *pointer };
    std::env::set_var("FOO", "BAR");
    let _y = unsafe { *pointer }; //~ ERROR dangling pointer was dereferenced
}
