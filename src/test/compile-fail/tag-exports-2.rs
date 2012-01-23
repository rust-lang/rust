// error-pattern:unresolved name: lovejoy
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
  let raleigh: irving = lovejoy;
}
