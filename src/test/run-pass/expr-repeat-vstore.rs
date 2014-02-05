#[feature(managed_boxes)];

pub fn main() {
    let v: ~[int] = ~[ 1, ..5 ];
    println!("{}", v[0]);
    println!("{}", v[1]);
    println!("{}", v[2]);
    println!("{}", v[3]);
    println!("{}", v[4]);
}
