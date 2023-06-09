// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

use alder::*;

mod alder {
    pub enum burnside { couch, davis }
    pub enum everett { flanders, glisan, hoyt }
    pub enum irving { johnson, kearney, lovejoy }
    pub enum marshall { northrup, overton }
}

pub fn main() {
  let _pettygrove: burnside = burnside::couch;
  let _quimby: everett = everett::flanders;
  let _raleigh: irving = irving::johnson;
  let _savier: marshall;
}
