// Breaks with values inside closures used to ICE (#66863)

fn main() {
    'some_label: loop {
        || break 'some_label ();
        //~^ ERROR: use of unreachable label `'some_label`
        //~| ERROR: `break` inside of a closure
    }
}
