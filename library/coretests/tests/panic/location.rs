use core::panic::Location;

// Note: Some of the following tests depend on the source location,
// so please be careful when editing this file.

mod file_a;
mod file_b;
mod file_c;

// A small shuffled set of locations for testing, along with their true order.
const LOCATIONS: [(usize, &'static Location<'_>); 9] = [
    (7, file_c::two()),
    (0, file_a::one()),
    (3, file_b::one()),
    (5, file_b::three()),
    (8, file_c::three()),
    (6, file_c::one()),
    (2, file_a::three()),
    (4, file_b::two()),
    (1, file_a::two()),
];

#[test]
fn location_const_caller() {
    const _CALLER_REFERENCE: &Location<'static> = Location::caller();
    const _CALLER: Location<'static> = *Location::caller();
}

#[test]
fn location_const_file() {
    const CALLER: &Location<'static> = Location::caller();
    const FILE: &str = CALLER.file();
    assert_eq!(FILE, file!());
}

#[test]
fn location_const_line() {
    const CALLER: &Location<'static> = Location::caller();
    const LINE: u32 = CALLER.line();
    assert_eq!(LINE, 38);
}

#[test]
fn location_const_column() {
    const CALLER: &Location<'static> = Location::caller();
    const COLUMN: u32 = CALLER.column();
    assert_eq!(COLUMN, 40);
}

#[test]
fn location_file_lifetime<'x>() {
    // Verify that the returned `&str`s lifetime is derived from the generic
    // lifetime 'a, not the lifetime of `&self`, when calling `Location::file`.
    // Test failure is indicated by a compile failure, not a runtime panic.
    let _: for<'a> fn(&'a Location<'x>) -> &'x str = Location::file;
}

#[test]
fn location_debug() {
    let f = format!("{:?}", Location::caller());
    assert!(f.contains(&format!("{:?}", file!())));
    assert!(f.contains("60"));
    assert!(f.contains("29"));
}

#[test]
fn location_eq() {
    for (i, a) in LOCATIONS {
        for (j, b) in LOCATIONS {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn location_ord() {
    let mut locations = LOCATIONS.clone();
    locations.sort_by_key(|(_o, l)| **l);
    for (correct, (order, _l)) in locations.iter().enumerate() {
        assert_eq!(correct, *order);
    }
}
