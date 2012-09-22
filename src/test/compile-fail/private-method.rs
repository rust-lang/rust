// error-pattern:method `nap` is private

mod kitties {
    #[legacy_exports];
struct cat {
  priv mut meows : uint,

  how_hungry : int,

}

impl cat {
    priv fn nap() { uint::range(1u, 10000u, |_i| false)}
}

fn cat(in_x : uint, in_y : int) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}
}

fn main() {
  let nyan : kitties::cat = kitties::cat(52u, 99);
  nyan.nap();
}
