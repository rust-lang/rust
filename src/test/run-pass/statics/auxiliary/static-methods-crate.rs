#![crate_name="static_methods_crate"]
#![crate_type = "lib"]

pub trait read: Sized {
    fn readMaybe(s: String) -> Option<Self>;
}

impl read for isize {
    fn readMaybe(s: String) -> Option<isize> {
        s.parse().ok()
    }
}

impl read for bool {
    fn readMaybe(s: String) -> Option<bool> {
        match &*s {
          "true" => Some(true),
          "false" => Some(false),
          _ => None
        }
    }
}

pub fn read<T:read>(s: String) -> T {
    match read::readMaybe(s) {
      Some(x) => x,
      _ => panic!("read panicked!")
    }
}
