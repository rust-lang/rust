// run-pass


pub fn main() { let x: Vec<isize> = Vec::new(); for _ in &x { panic!("moop"); } }
