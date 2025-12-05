//@ run-pass
fn main() {
    let s: &[isize] = &[0, 1, 2, 3, 4];
    let ss: &&[isize] = &s;
    let sss: &&&[isize] = &ss;

    println!("{:?}", &s[..3]);
    println!("{:?}", &ss[3..]);
    println!("{:?}", &sss[2..4]);
}
