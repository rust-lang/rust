fn main() {
    'aaaaab: loop {
        || {
            loop { continue 'aaaaaa }
            //~^ ERROR use of undeclared label `'aaaaaa`
        };

    }
}
