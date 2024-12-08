fn main() {

    let y: Box<isize> = 42.into();
    let mut x: Box<isize>;

    loop {
        println!("{}", y);
        loop {
            loop {
                loop {
                    x = y; //~ ERROR use of moved value
                    x.clone();
                }
            }
        }
    }
}
