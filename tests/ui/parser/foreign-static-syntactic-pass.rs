// Syntactically, a foreign static may have a body.

// check-pass

fn main() {}

#[cfg(FALSE)]
extern "C" {
    static X: u8;
    static mut Y: u8;
}
