// rustfmt-spaces_around_ranges: false
// Spaces around ranges

fn main() {
    let lorem = 0 .. 10;
    let ipsum = 0 ..= 10;

    match lorem {
        1 .. 5 => foo(),
        _ => bar,
    }

    match lorem {
        1 ..= 5 => foo(),
        _ => bar,
    }

    match lorem {
        1 ... 5 => foo(),
        _ => bar,
    }
}
