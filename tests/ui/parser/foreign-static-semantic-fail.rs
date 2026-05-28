// Syntactically, a foreign static may not have a body.

fn main() {}

extern "C" {
    static X: u8 = 0; //~ ERROR incorrect `static` inside `extern` block
    static mut Y: u8 = 0; //~ ERROR incorrect `static` inside `extern` block
}
