struct Bug([u8; panic!("panic")]); //~ ERROR evaluation of constant value failed

fn main() {}
