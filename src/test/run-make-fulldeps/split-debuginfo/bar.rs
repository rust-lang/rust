pub struct Bar {
    x: u32,
}

impl Bar {
    pub fn print(&self) {
        println!("{}", self.x);
    }
}

pub fn make_bar(x: u32) -> Bar {
    Bar { x }
}
