// Syntactically, a foreign static may have a body.

// check-pass

fn main() {}

#[cfg(FALSE)]
extern {
    static X: u8 = 0;
    static mut Y: u8 = 0;
}
