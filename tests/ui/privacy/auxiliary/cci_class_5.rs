pub mod kitties {
    pub struct cat {
        meows : usize,
        pub how_hungry : isize,
    }

    impl cat {
        fn nap(&self) {}
    }

    pub fn cat(in_x : usize, in_y : isize) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y
        }
    }
}
