// There isn't a great way to test feature(nll), since it just disables migrate
// mode and changes some error messages.

// FIXME(Centril): This test is probably obsolete now and `nll` should become
// `accepted`.

// Don't use compare-mode=nll, since that turns on NLL.
// ignore-compare-mode-nll
// ignore-compare-mode-polonius

fn main() {
    let mut x = (33, &0);

    let m = &mut x;
    let p = &*x.1;
    //~^ ERROR cannot borrow
    m;
}
