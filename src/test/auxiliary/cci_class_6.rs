mod kitties {

struct cat<U> {
  priv {
    mut info : ~[U],
    mut meows : uint,
  }

  how_hungry : int,
}

    impl<U> cat<U> {
        fn speak<T>(stuff: ~[T]) {
            self.meows += stuff.len();
        }
        fn meow_count() -> uint { self.meows }
    }

fn cat<U>(in_x : uint, in_y : int, -in_info: ~[U]) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        info: in_info
    }
}


}
