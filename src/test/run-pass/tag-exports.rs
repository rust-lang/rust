import alder::*;

mod alder {
  export burnside;
  export couch;
  export everett;
  export flanders;
  export irving;
  export johnson;
  export kearney;
  export marshall;

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
