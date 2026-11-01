//@ check-pass

unsafe extern "C" {
    #[diagnostic::c_buffer_length(buffer, length)]
    //~^ WARN unknown diagnostic attribute
    fn c_fn(buffer: *const u8, length: usize);
}

fn main() {}
