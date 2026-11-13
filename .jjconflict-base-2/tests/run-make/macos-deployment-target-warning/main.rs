#![warn(linker_info, linker_messages)]
unsafe extern "C" {
    safe fn foo();
}

fn main() {
    foo();
}
