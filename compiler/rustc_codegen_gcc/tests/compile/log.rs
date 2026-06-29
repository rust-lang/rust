// Compiler:

extern "C" {
    fn log(message_data: u32, message_size: u32);
}

pub fn main() {
    let message = "Hello, world!";
    unsafe {
        log(message.as_ptr() as u32, message.len() as u32);
    }
}
