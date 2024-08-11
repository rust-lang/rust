#![deny(instantly_dangling_pointer)]

fn main() {
    vec![0u8].as_ptr(); //~ ERROR [instantly_dangling_pointer]
    vec![0u8].as_mut_ptr(); //~ ERROR [instantly_dangling_pointer]
}
