// run-pass
#![allow(dead_code)]

enum Animal {
    Cat = 0,
    Dog = 1,
    Horse = 2,
    Snake = 3,
}

enum Hero {
    Batman = -1,
    Superman = -2,
    Ironman = -3,
    Spiderman = -4
}

pub fn main() {
    let pet: Animal = Animal::Snake;
    let hero: Hero = Hero::Superman;
    assert_eq!(pet as usize, 3);
    assert_eq!(hero as isize, -2);
}
