#![feature(register_tool)]
#![register_tool(lint)]

unsafe extern "C" {
    #[lint::c_buffer_length(buffer, length)]
    //~^ ERROR the `lint::c_buffer_length` attribute is an experimental feature
    fn c_fn(buffer: *const u8, length: usize);
}

fn main() {}
