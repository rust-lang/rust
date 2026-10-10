// Regression test for #69409

struct Cat {
    meows : usize,
    how_hungry : isize,
}

impl Cat {
    pub fn speak(&mut self) { self.meows += 1; }
}

fn cat(in_x : usize, in_y : isize) -> Cat {
    Cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
    let nyan : Cat = cat(52, 99);
    nyan.speak = || println!("meow"); //~ ERROR attempted to take value of method
    nyan.speak += || println!("meow"); //~ ERROR attempted to take value of method
}
