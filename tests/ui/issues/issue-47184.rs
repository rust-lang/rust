fn main() {
    let _vec: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR temporary value dropped while borrowed [E0716]
}
