// -*- rust -*-

fn main() {
    let u32 word = (200000 as u32);
    word = word - (1 as u32);
    check(word == (199999 as u32));
}

