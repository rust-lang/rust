//@ build-pass (FIXME(62277): could be check-pass?)
//@ pp-exact - Make sure we print all the attributes

#![feature(rustc_attrs)]

#[rustc_dummy]
trait Frobable {
    #[rustc_dummy]
    fn frob(&self);
    #[rustc_dummy]
    fn defrob(&self);
}

#[rustc_dummy]
impl Frobable for isize {
    #[rustc_dummy]
    fn frob(&self) {
        #![rustc_dummy]
    }

    #[rustc_dummy]
    fn defrob(&self) {
        #![rustc_dummy]
    }
}

fn main() {}
