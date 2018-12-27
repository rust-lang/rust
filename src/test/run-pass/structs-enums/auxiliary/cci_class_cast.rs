pub mod kitty {
    use std::fmt;

    pub struct cat {
      meows : usize,
      pub how_hungry : isize,
      pub name : String,
    }

    impl fmt::Display for cat {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    impl cat {
        fn meow(&mut self) {
            println!("Meow");
            self.meows += 1;
            if self.meows % 5 == 0 {
                self.how_hungry += 1;
            }
        }

    }

    impl cat {
        pub fn speak(&mut self) { self.meow(); }

        pub fn eat(&mut self) -> bool {
            if self.how_hungry > 0 {
                println!("OM NOM NOM");
                self.how_hungry -= 2;
                return true;
            }
            else {
                println!("Not hungry!");
                return false;
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
