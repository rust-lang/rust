fn main() {
    while i < days.len() {
        #![allow(clippy::indexing_slicing)]
        w |= days[i].bit_value();
        i += 1;
    }

    for i in 0..days.len() {
        #![allow(clippy::indexing_slicing)]
        w |= days[i].bit_value();
        i += 1;
    }

    loop {
        #![allow(clippy::indexing_slicing)]
        w |= days[i].bit_value();
        i += 1;
    }
}
