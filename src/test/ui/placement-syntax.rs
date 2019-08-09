fn main() {
    let x = -5;
    if x<-1 { //~ ERROR expected `{`, found `<-`
        println!("ok");
    }
}
