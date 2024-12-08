// rustfmt-spaces_around_ranges: true
// Spaces around ranges

fn main() {
    let lorem = 0..10;
    let ipsum = 0..=10;

    match lorem {
        1..5 => foo(),
        _ => bar,
    }

    match lorem {
        1..=5 => foo(),
        _ => bar,
    }

    match lorem {
        1...5 => foo(),
        _ => bar,
    }
}

fn half_open() {
    match [5..4, 99..105, 43..44] {
        [_, 99.., _] => {}
        [_, ..105, _] => {}
        _ => {}
    };

    if let ..=5 = 0 {}
    if let ..5 = 0 {}
    if let 5.. = 0 {}
}
