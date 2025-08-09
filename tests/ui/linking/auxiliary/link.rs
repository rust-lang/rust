// Dummy linker implementation simulating __fastfail termination
// See <https://devblogs.microsoft.com/oldnewthing/20190108-00/?p=100655>
fn main() {
    const STATUS_STACK_BUFFER_OVERRUN: i32 = 0xC0000409u32 as i32;
    std::process::exit(STATUS_STACK_BUFFER_OVERRUN);
}
