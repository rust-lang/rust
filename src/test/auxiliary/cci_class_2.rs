mod kitties {

struct cat {
  priv mut meows : uint,

  how_hungry : int,

}

    impl cat {
        fn speak() {}
    }
    fn cat(in_x : uint, in_y : int) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }

}
