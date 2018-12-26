trait Bar {}

impl Bar for i32 {
    const BAR: bool = true; //~ ERROR E0438
}

fn main () {
}
