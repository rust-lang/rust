#[link(name = "static_methods_crate",
       vers = "0.1")];

#[crate_type = "lib"];

export read, readMaybe;

trait read {
    static fn readMaybe(s: ~str) -> option<self>;
}

impl of read for int {
    static fn readMaybe(s: ~str) -> option<int> {
        int::from_str(s)
    }
}

impl of read for bool {
    static fn readMaybe(s: ~str) -> option<bool> {
        match s {
          ~"true" => some(true),
          ~"false" => some(false),
          _ => none
        }
    }
}

fn read<T: read copy>(s: ~str) -> T {
    match readMaybe(s) {
      some(x) => x,
      _ => fail ~"read failed!"
    }
}
