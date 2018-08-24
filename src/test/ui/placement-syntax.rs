fn main() {
    let x = -5;
    if x<-1 {
    //~^ ERROR emplacement syntax is obsolete
        println!("ok");
    }
}
