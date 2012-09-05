mod kitties {

struct cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  fn speak() { self.meows += 1u; }
  fn meow_count() -> uint { self.meows }

}

    fn cat(in_x : uint, in_y : int) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }


}
