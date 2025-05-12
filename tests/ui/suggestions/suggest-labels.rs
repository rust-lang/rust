#[allow(unreachable_code, unused_labels)]
fn main() {
    'foo: loop {
        break 'fo; //~ ERROR use of undeclared label
    }

    'bar: loop {
        continue 'bor; //~ ERROR use of undeclared label
    }

    'longlabel: loop {
        'longlabel1: loop {
            break 'longlable; //~ ERROR use of undeclared label
        }
    }
}
