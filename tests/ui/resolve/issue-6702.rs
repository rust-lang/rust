struct Monster {
    damage: isize
}


fn main() {
    let _m = Monster();
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `Monster`
}
