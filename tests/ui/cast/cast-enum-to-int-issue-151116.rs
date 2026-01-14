#[repr(u8)]
enum Priority {
    High = 255,
    Normal = 127,
    Low = 1,
}

fn main() {
    let priority = &Priority::Normal;
    let priority = priority as u8; //~ ERROR casting `&Priority` as `u8` is invalid
}
