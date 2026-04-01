fn vindictive() -> bool { true }

fn perfidy() -> (i32, i32) {
    vindictive() //~ ERROR expected function, found `bool`
    (1, 2)
}

fn main() {}
