//@ignore-target: windows # Windows does not have a global environ list that the program can access directly

fn get_environ() -> *const *const u8 {
    extern "C" {
        static mut environ: *const *const u8;
    }
    unsafe { environ }
}

fn main() {
    let pointer = get_environ();
    let _x = unsafe { *pointer };
    std::env::set_var("FOO", "BAR");
    let _y = unsafe { *pointer }; //~ ERROR: has been freed
}
