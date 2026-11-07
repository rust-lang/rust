fn main() {
    'a: loop {
        || {
            //~^ ERROR mismatched types
            loop { break 'a; } //~ ERROR E0767
        }
    }
}
