#![deny(overlapping_range_endpoints)]

macro_rules! m {
    ($s:expr, $t1:pat, $t2:pat) => {
        match $s {
            $t1 => {}
            $t2 => {}
            _ => {}
        }
    };
}

fn main() {
    m!(0u8, 20..=30, 30..=40); //~ ERROR multiple patterns overlap on their endpoints
    m!(0u8, 30..=40, 20..=30); //~ ERROR multiple patterns overlap on their endpoints
    m!(0u8, 20..=30, 31..=40);
    m!(0u8, 20..=30, 29..=40);
    m!(0u8, 20..30, 29..=40); //~ ERROR multiple patterns overlap on their endpoints
    m!(0u8, 20..30, 28..=40);
    m!(0u8, 20..30, 30..=40);
    m!(0u8, 20..=30, 30..=30);
    m!(0u8, 20..=30, 30..=31); //~ ERROR multiple patterns overlap on their endpoints
    m!(0u8, 20..=30, 29..=30);
    m!(0u8, 20..=30, 20..=20);
    m!(0u8, 20..=30, 20..=21);
    m!(0u8, 20..=30, 19..=20); //~ ERROR multiple patterns overlap on their endpoints
    m!(0u8, 20..=30, 20);
    m!(0u8, 20..=30, 25);
    m!(0u8, 20..=30, 30);
    m!(0u8, 20..30, 29);
    m!(0u8, 20, 20..=30);
    m!(0u8, 25, 20..=30);
    m!(0u8, 30, 20..=30);

    match 0u8 {
        0..=10 => {}
        20..=30 => {}
        10..=20 => {}
        //~^ ERROR multiple patterns overlap on their endpoints
        //~| ERROR multiple patterns overlap on their endpoints
        _ => {}
    }
    match (0u8, true) {
        (0..=10, true) => {}
        (10..20, true) => {} //~ ERROR multiple patterns overlap on their endpoints
        (10..20, false) => {}
        _ => {}
    }
    match (true, 0u8) {
        (true, 0..=10) => {}
        (true, 10..20) => {} //~ ERROR multiple patterns overlap on their endpoints
        (false, 10..20) => {}
        _ => {}
    }
    match Some(0u8) {
        Some(0..=10) => {}
        Some(10..20) => {} //~ ERROR multiple patterns overlap on their endpoints
        _ => {}
    }

    // The lint has false negatives when we skip some cases because of relevancy.
    match (true, true, 0u8) {
        (true, _, 0..=10) => {}
        (_, true, 10..20) => {}
        _ => {}
    }
}
