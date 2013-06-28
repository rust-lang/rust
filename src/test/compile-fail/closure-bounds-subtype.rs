
fn take_any(_: &fn:()) {
}

fn take_copyable(_: &fn:Copy()) {
}

fn take_copyable_owned(_: &fn:Copy+Send()) {
}

fn take_const_owned(_: &fn:Freeze+Send()) {
}

fn give_any(f: &fn:()) {
    take_any(f);
    take_copyable(f); //~ ERROR expected bounds `Copy` but found no bounds
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Send` but found no bounds
}

fn give_copyable(f: &fn:Copy()) {
    take_any(f);
    take_copyable(f);
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Send` but found bounds `Copy`
}

fn give_owned(f: &fn:Send()) {
    take_any(f);
    take_copyable(f); //~ ERROR expected bounds `Copy` but found bounds `Send`
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Send` but found bounds `Send`
}

fn give_copyable_owned(f: &fn:Copy+Send()) {
    take_any(f);
    take_copyable(f);
    take_copyable_owned(f);
    take_const_owned(f); //~ ERROR expected bounds `Send+Freeze` but found bounds `Copy+Send`
}

fn main() {}
