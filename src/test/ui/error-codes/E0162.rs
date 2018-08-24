struct Irrefutable(i32);

fn main() {
    let irr = Irrefutable(0);
    if let Irrefutable(x) = irr { //~ ERROR E0162
        println!("{}", x);
    }
}
