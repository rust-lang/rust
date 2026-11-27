use std::io::Read;

fn f<T: Read, U, Read>() {} //~ ERROR expected trait, found type parameter `Read`

fn main() {}
