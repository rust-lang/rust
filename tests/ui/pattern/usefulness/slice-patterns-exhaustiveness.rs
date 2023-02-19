fn main() {
    let s: &[bool] = &[true; 0];
    let s1: &[bool; 1] = &[false; 1];
    let s2: &[bool; 2] = &[false; 2];
    let s3: &[bool; 3] = &[false; 3];
    let s10: &[bool; 10] = &[false; 10];

    match s2 {
        //~^ ERROR match is non-exhaustive [E0004]
        [true, .., true] => {}
    }
    match s3 {
        //~^ ERROR match is non-exhaustive [E0004]
        [true, .., true] => {}
    }
    match s10 {
        //~^ ERROR match is non-exhaustive [E0004]
        [true, .., true] => {}
    }

    match s1 {
        [true, ..] => {}
        [.., false] => {}
    }
    match s2 {
        //~^ ERROR match is non-exhaustive [E0004]
        [true, ..] => {}
        [.., false] => {}
    }
    match s3 {
        //~^ ERROR match is non-exhaustive [E0004]
        [true, ..] => {}
        [.., false] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [true, ..] => {}
        [.., false] => {}
    }

    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [_] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [true, ..] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [_] => {}
        [true, ..] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [_] => {}
        [.., true] => {}
    }

    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [_] => {}
        [_, _] => {}
        [.., false] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        [] => {}
        [_] => {}
        [_, _] => {}
        [false, .., false] => {}
    }

    const CONST: &[bool] = &[true];
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        &[true] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        CONST => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        CONST => {}
        &[false] => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        &[false] => {}
        CONST => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        &[] => {}
        CONST => {}
    }
    match s {
        //~^ ERROR match is non-exhaustive [E0004]
        &[] => {}
        CONST => {}
        &[_, _, ..] => {}
    }
    match s {
        [] => {}
        [false] => {}
        CONST => {}
        [_, _, ..] => {}
    }
    const CONST1: &[bool; 1] = &[true];
    match s1 {
        //~^ ERROR match is non-exhaustive [E0004]
        CONST1 => {}
    }
    match s1 {
        CONST1 => {}
        [false] => {}
    }
}
