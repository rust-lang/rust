mod kitties {

struct cat {
  priv {
    let mut meows : uint;
      fn nap() { for uint::range(1u, 10000u) |_i|{}}
  }

  let how_hungry : int;

}

    fn cat(in_x : uint, in_y : int) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }

}