fn main () {
    'a: loop {
        || {
            loop { break 'a; } //~ ERROR E0767
        }
    }
}
