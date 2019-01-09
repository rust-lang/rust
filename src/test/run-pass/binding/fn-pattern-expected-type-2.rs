// run-pass
pub fn main() {
    let v : &[(isize,isize)] = &[ (1, 2), (3, 4), (5, 6) ];
    for &(x, y) in v {
        println!("{}", y);
        println!("{}", x);
    }
}
