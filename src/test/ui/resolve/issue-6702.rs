struct Monster {
    damage: isize
}


fn main() {
    let _m = Monster(); //~ ERROR expected function, found struct `Monster`
}
