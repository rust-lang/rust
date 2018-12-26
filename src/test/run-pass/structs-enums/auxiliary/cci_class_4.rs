pub mod kitties {
    pub struct cat {
        meows : usize,

        pub how_hungry : isize,
        pub name : String,
    }

    impl cat {
        pub fn speak(&mut self) { self.meow(); }

        pub fn eat(&mut self) -> bool {
            if self.how_hungry > 0 {
                println!("OM NOM NOM");
                self.how_hungry -= 2;
                return true;
            } else {
                println!("Not hungry!");
                return false;
            }
        }
    }

    impl cat {
        pub fn meow(&mut self) {
            println!("Meow");
            self.meows += 1;
            if self.meows % 5 == 0 {
                self.how_hungry += 1;
            }
        }
    }

    pub fn cat(in_x : usize, in_y : isize, in_name: String) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y,
            name: in_name
        }
    }
}
