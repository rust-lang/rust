pub mod kitties {
    pub struct cat {
        meows : usize,

        pub how_hungry : isize,
    }

    impl cat {
        pub fn speak(&mut self) { self.meows += 1; }
        pub fn meow_count(&mut self) -> usize { self.meows }
    }

    pub fn cat(in_x : usize, in_y : isize) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }
}
