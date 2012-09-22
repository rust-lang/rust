#[link(name = "static_methods_crate",
       vers = "0.1")];

#[crate_type = "lib"];
#[legacy_exports];
export read, readMaybe;

trait read {
    static fn readMaybe(s: ~str) -> Option<self>;
}

impl int: read {
    static fn readMaybe(s: ~str) -> Option<int> {
        int::from_str(s)
    }
}

impl bool: read {
    static fn readMaybe(s: ~str) -> Option<bool> {
        match s {
          ~"true" => Some(true),
          ~"false" => Some(false),
          _ => None
        }
    }
}

fn read<T: read Copy>(s: ~str) -> T {
    match readMaybe(s) {
      Some(x) => x,
      _ => fail ~"read failed!"
    }
}
