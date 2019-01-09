// run-pass
#![allow(unused_attributes)]
#![allow(non_camel_case_types)]

// pp-exact - Make sure we print all the attributes
// pretty-expanded FIXME #23616

#![feature(custom_attribute)]

#[frobable]
trait frobable {
    #[frob_attr]
    fn frob(&self);
    #[defrob_attr]
    fn defrob(&self);
}

#[int_frobable]
impl frobable for isize {
    #[frob_attr1]
    fn frob(&self) {
        #![frob_attr2]
    }

    #[defrob_attr1]
    fn defrob(&self) {
        #![defrob_attr2]
    }
}

pub fn main() { }
