pub struct Example { a: i32 }

impl Example {
    fn is_pos(&self) -> bool { self.a > 0 }
}

fn one() -> i32 { 1 }

fn main() {
    if Example { a: one(), }.is_pos() { //~ ERROR invalid struct literal
        println!("Positive!");
    }
}
