
// Replace every top-level comma with `|`, preserving commas inside tuple patterns.

fn matches_pair(pair: (u8, u8)) -> bool {
    match pair {
        (0, 1), (2, 3), (4, 5) => true,
        //~^ ERROR unexpected `,` in pattern
        _ => false,
    }
}

fn main() {
    assert!(matches_pair((0, 1)));
    assert!(!matches_pair((0, 5)));
}
