
fn take_any(_: &fn:()) {
}

fn take_copyable(_: &fn:Copy()) {
}

fn take_copyable_owned(_: &fn:Copy+Owned()) {
}

fn take_const_owned(_: &fn:Const+Owned()) {
}

fn give_any(f: &fn:()) {
    take_any(f);
    take_copyable(f); //~ ERROR expected bounds `Copy` but found no bounds
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Owned` but found no bounds
}

fn give_copyable(f: &fn:Copy()) {
    take_any(f);
    take_copyable(f);
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Owned` but found bounds `Copy`
}

fn give_owned(f: &fn:Owned()) {
    take_any(f);
    take_copyable(f); //~ ERROR expected bounds `Copy` but found bounds `Owned`
    take_copyable_owned(f); //~ ERROR expected bounds `Copy+Owned` but found bounds `Owned`
}

fn give_copyable_owned(f: &fn:Copy+Owned()) {
    take_any(f);
    take_copyable(f);
    take_copyable_owned(f);
    take_const_owned(f); //~ ERROR expected bounds `Owned+Const` but found bounds `Copy+Owned`
}

fn main() {}
