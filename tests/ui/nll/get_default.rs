// Basic test for free regions in the NLL code. This test ought to
// report an error due to a reborrowing constraint.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

struct Map {
}

impl Map {
    fn get(&self) -> Option<&String> { None }
    fn set(&mut self, v: String) { }
}

fn ok(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                return v;
            }
            None => {
                map.set(String::new()); // Ideally, this would not error.
                //[nll]~^ ERROR borrowed as immutable
            }
        }
    }
}

fn err(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                map.set(String::new()); // We always expect an error here.
                //~^ ERROR borrowed as immutable
                return v;
            }
            None => {
                map.set(String::new()); // Ideally, this would not error.
                //[nll]~^ ERROR borrowed as immutable
            }
        }
    }
}

fn main() { }
