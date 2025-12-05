fn main() {
    let x = -5;
    if x<-1 { //~ ERROR unexpected token: `<-`
        println!("ok");
    }
}
