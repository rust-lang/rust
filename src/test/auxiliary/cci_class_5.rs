mod kitties {

    pub struct cat {
        priv mut meows : uint,
        how_hungry : int,
    }

    impl cat {
      priv fn nap() { for uint::range(1, 10000u) |_i|{}}
    }

    pub fn cat(in_x : uint, in_y : int) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }

}