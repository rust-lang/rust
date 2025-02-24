fn main() {
    let mut s = String::from("hello");
    let mut ref_s = &mut s;

    ref_s = &mut String::from("world"); //~ ERROR temporary value dropped while borrowed [E0716]

    print!("r1 = {}", ref_s);
}
