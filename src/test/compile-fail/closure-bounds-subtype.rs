
fn take_any(_: &fn:()) {
}

fn take_const_owned(_: &fn:Freeze+Send()) {
}

fn give_any(f: &fn:()) {
    take_any(f);
}

fn give_owned(f: &fn:Send()) {
    take_any(f);
    take_const_owned(f); //~ ERROR expected bounds `Send+Freeze` but found bounds `Send`
}

fn main() {}
