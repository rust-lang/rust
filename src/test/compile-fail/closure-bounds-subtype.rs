
fn take_any(_: ||:) {
}

fn take_const_owned(_: ||:Freeze+Send) {
}

fn give_any(f: ||:) {
    take_any(f);
}

fn give_owned(f: ||:Send) {
    take_any(f);
    take_const_owned(f); //~ ERROR expected bounds `Send+Freeze` but found bounds `Send`
}

fn main() {}
