enum Enum { A, B, C }

fn func(inbounds: &Enum, array: &[i16; 3]) -> i16 {
    array[*inbounds as usize]
    //~^ ERROR [E0507]
}

fn main() {}
