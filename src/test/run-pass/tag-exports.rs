import alder::*;

mod alder {
  export burnside;
  export everett::{flanders};
  export irving::{johnson, kearney};
  export marshall::{};

  enum burnside { couch, davis }
  enum everett { flanders, glisan, hoyt }
  enum irving { johnson, kearney, lovejoy }
  enum marshall { northrup, overton }

}

fn main() {
  let _pettygrove: burnside = couch;
  let _quimby: everett = flanders;
  let _raleigh: irving = johnson;
  let _savier: marshall;
}
