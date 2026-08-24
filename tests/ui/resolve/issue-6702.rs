struct Monster {
    damage: isize
}


fn main() {
    let _m = Monster();
    //~^ ERROR cannot find function, tuple struct or tuple variant `Monster` in this scope
}
