//@ run-pass
#![allow(non_camel_case_types)]
#![allow(dead_code)]

trait noisy {
    fn speak(&mut self);
}

#[derive(Clone)]
struct cat {
    meows : usize,

    how_hungry : isize,
    name : String,
}

impl cat {
    fn meow(&mut self) {
        println!("Meow");
        self.meows += 1_usize;
        if self.meows % 5_usize == 0_usize {
            self.how_hungry += 1;
        }
    }
}

impl cat {
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

impl noisy for cat {
    fn speak(&mut self) { self.meow(); }
}

fn cat(in_x : usize, in_y : isize, in_name: String) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name.clone()
    }
}


fn make_speak<C:noisy>(mut c: C) {
    c.speak();
}

pub fn main() {
    let mut nyan = cat(0_usize, 2, "nyan".to_string());
    nyan.eat();
    assert!(!nyan.eat());
    for _ in 1_usize..10_usize {
        make_speak(nyan.clone());
    }
}
