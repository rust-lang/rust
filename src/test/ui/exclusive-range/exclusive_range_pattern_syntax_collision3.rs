#![feature(exclusive_range_pattern)]

fn main() {
    match [5..4, 99..105, 43..44] {
        [..9, 99..100, _] => {}, //~ ERROR expected one of `,` or `]`, found `9`
        _ => {},
    }
}
