//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: -Zpolonius=off
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

use std::cell::Cell;

struct RequiresStatic<T: 'static>(Option<T>);
struct Outlives<'a, 'b: 'a>(Cell<&'a ()>, Option<&'b str>);

fn static_bound() {
    let text = String::from("temporary");
    let mut value = RequiresStatic(None);
    value.0 = Some(&text);
    //~^ ERROR `text` does not live long enough
}

fn outlives_bound(fixed: Cell<&'static ()>) {
    let text = String::from("temporary");
    let mut value = Outlives(fixed, None);
    value.1 = Some(&text);
    //~^ ERROR `text` does not live long enough
}

fn main() {}
