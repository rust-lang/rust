extern "C" {
    static error_message_count: u32;
}

pub static BAZ: u32 = *&error_message_count;
//~^ ERROR use of extern static is unsafe and requires
//~| ERROR could not evaluate static initializer

fn main() {}
