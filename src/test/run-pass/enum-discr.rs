enum Animal {
    Cat = 0u,
    Dog = 1u,
    Horse = 2u,
    Snake = 3u
}

enum Hero {
    Batman = -1,
    Superman = -2,
    Ironman = -3,
    Spiderman = -4
}

pub fn main() {
    let pet: Animal = Snake;
    let hero: Hero = Superman;
    assert!(pet as uint == 3);
    assert!(hero as int == -2);
}
