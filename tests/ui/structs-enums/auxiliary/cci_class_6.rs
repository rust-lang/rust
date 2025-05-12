pub mod kitties {

    pub struct cat<U> {
        info : Vec<U> ,
        meows : usize,

        pub how_hungry : isize,
    }

    impl<U> cat<U> {
        pub fn speak<T>(&mut self, stuff: Vec<T> ) {
            self.meows += stuff.len();
        }

        pub fn meow_count(&mut self) -> usize { self.meows }
    }

    pub fn cat<U>(in_x : usize, in_y : isize, in_info: Vec<U> ) -> cat<U> {
        cat {
            meows: in_x,
            how_hungry: in_y,
            info: in_info
        }
    }
}
