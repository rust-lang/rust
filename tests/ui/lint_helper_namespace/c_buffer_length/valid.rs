//@ check-pass

#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

#[lint::c_buffer_length(buffer, length)]
fn rust_fn(buffer: *const u8, length: usize) {}

#[lint::c_buffer_length(buffer, length)]
fn generic<T>(buffer: *mut T, length: isize) {}

type Buffer = *const u8;
type Length = u32;

#[lint::c_buffer_length(r#type, r#match,)]
fn aliases(r#match: Length, r#type: Buffer) {}

unsafe extern "C" {
    #[lint::c_buffer_length(buffer, length)]
    fn c_fn(buffer: *const u8, length: usize);

    #[lint::c_buffer_length(buffer, length)]
    fn c_mut_fn(length: i32, buffer: *mut u8);
}

fn main() {}
