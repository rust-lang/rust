use std::mem::transmute;

fn get_flag<const FlagSet: bool, const ShortName: char>() -> Option<char> {
  if FlagSet {
    Some(ShortName)
  } else {
    None
  }
}

union CharRaw {
  byte: u8,
  character: char,
}

union BoolRaw {
  byte: u8,
  boolean: bool,
}

const char_raw: CharRaw = CharRaw { byte: 0xFF };
const bool_raw: BoolRaw = BoolRaw { byte: 0x42 };

fn main() {
  // Test that basic cases don't work
  assert!(get_flag::<true, 'c'>().is_some());
  assert!(get_flag::<false, 'x'>().is_none());
  get_flag::<false, 0xFF>();
  //~^ ERROR mismatched types
  get_flag::<7, 'c'>();
  //~^ ERROR mismatched types
  get_flag::<42, 0x5ad>();
  //~^ ERROR mismatched types
  //~| ERROR mismatched types


  get_flag::<false, { unsafe { char_raw.character } }>();
  //~^ ERROR it is undefined behavior
  get_flag::<{ unsafe { bool_raw.boolean } }, 'z'>();
  //~^ ERROR it is undefined behavior
  get_flag::<{ unsafe { bool_raw.boolean } }, { unsafe { char_raw.character } }>();
  //~^ ERROR it is undefined behavior
  //~| ERROR it is undefined behavior
}
