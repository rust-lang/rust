// -*- rust -*-

fn main() {
    let u32 word = u32(200000);
    word = word - u32(1);
    check(word == u32(199999));
}

