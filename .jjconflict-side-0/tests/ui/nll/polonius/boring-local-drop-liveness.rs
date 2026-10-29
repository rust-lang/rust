// Polonius requires liveness for some locals that NLL leaves "boring": locals
// containing some region that outlives a universal region but is not universal itself.
//
// This test checks that we correctly compute liveness for NLL-boring locals that
// are live only because of their drop.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius_next
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius_next] compile-flags: -Zpolonius=next

struct D<'a>(&'a String);

impl<'a> Drop for D<'a> {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

fn assigning_under_a_live_destructor<'a>(
    x: &'a mut String,
    slot: &mut Option<&'a String>,
    y: &'a String,
) {
    let mut d = D(y);
    *slot = Some(d.0);
    d = D(&*x);
    *x = String::new(); //~ ERROR cannot assign to `*x` because it is borrowed
}

fn dropping_under_a_live_destructor<'a>(x: &'a String, slot: &mut Option<&'a String>) {
    let mut d = D(x);
    *slot = Some(d.0);
    let local = String::from("gone");
    d = D(&local); //~ ERROR `local` does not live long enough
}

fn main() {}
