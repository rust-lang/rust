struct Bug([u8; panic!("panic")]); //~ ERROR panic

fn main() {}
