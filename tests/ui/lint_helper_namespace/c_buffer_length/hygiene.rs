//@ check-pass

#![feature(decl_macro, lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

macro declare($buffer:ident) {
    #[lint::c_buffer_length($buffer, len)]
    fn abc($buffer: *mut u8, len: isize) {}
}

declare!(len);

fn main() {}
