#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

#[lint::c_buffer_length(buffer, length)]
//~^ ERROR attribute cannot be used on
struct Misplaced;

impl Misplaced {
    #[lint::c_buffer_length(buffer, length)]
    //~^ ERROR attribute cannot be used on
    fn method(buffer: *const u8, length: usize) {}
}

unsafe extern "C" {
    #[lint::c_buffer_length(buffer, length)]
    //~^ ERROR attribute cannot be used on
    static FOREIGN: u8;
}

fn main() {}
