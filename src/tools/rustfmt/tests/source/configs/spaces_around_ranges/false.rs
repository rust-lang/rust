// rustfmt-spaces_around_ranges: false
// Spaces around ranges

fn main() {
    let lorem = 0 .. 10;
    let ipsum = 0 ..= 10;

    match lorem {
        1 .. 5 => foo(),
        1. .. 5. => (),
        _ => bar,
    }

    match lorem {
        1 ..= 5 => foo(),
        1. ..= 5. => (),
        _ => bar,
    }

    match lorem {
        1 ... 5 => foo(),
        1. ... 0.5 => foo(),
        _ => bar,
    }
}

fn half_open() {
    match [5 .. 4, 99 .. 105, 43 .. 44] {
        [_, 99 .., _] => {}
        [_, .. 105, _] => {}
        [_, 1. .., _] => {}
        _ => {}
    };

    if let ..=   5 = 0 {}
    if let .. 5 = 0 {}
    // For now `.. .5` fails parsing with `float literals must have an integer part`
    if let .. 0.5 = 0 {}
    if let 5 .. = 0 {}
    if let 5. .. = 0 {}
}

fn pattern_in_function_parameters_exactly_max_width_before_space__(2. .. 10.: std::ops::Range<f32>) {

}
