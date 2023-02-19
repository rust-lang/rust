fn main() {
    match 0 { 1 => () } //~ ERROR match is non-exhaustive [E0004]
    match 0 { 0 if false => () } //~ ERROR match is non-exhaustive [E0004]
}
