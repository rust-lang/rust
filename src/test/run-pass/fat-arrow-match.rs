enum color {
    red,
    green,
    blue
}

pub fn main() {
    println!("{}", match color::red {
        color::red => { 1 }
        color::green => { 2 }
        color::blue => { 3 }
    });
}
