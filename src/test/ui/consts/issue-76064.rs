struct Bug([u8; panic!(1)]); //~ ERROR panicking in constants is unstable

fn main() {}
