// xfail-test
mod alder {
  export burnside;
  export everett::{flanders};
  export irving::{johnson, kearney};
  export marshall::{};

  tag burnside { couch, davis }
  tag everett { flanders, glisan, hoyt }
  tag irving { johnson, kearney, lovejoy }
  tag marshall { northrup, overton }
}

import alder::*;

fn main() {
  let pettygrove: burnside = couch;
  let quimby: everett = flanders;
  let raleigh: irving = johnson;
  let savier: marshall;
}