fn main() {
    'a: loop {
        || {
            loop { continue 'a }
            //~^ ERROR use of unreachable label `'a`
        };

    }
}
