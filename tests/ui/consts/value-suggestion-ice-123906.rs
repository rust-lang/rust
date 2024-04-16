fn as_chunks<const N: usize>() -> [u8; N] {
    loop {
        break;
        //~^ ERROR mismatched types
    }
}

fn main() {}
