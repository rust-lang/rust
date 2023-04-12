fn main() {
    if 1 > 2 {
        println!("Hello.");
   } elif 2 > 1 {
       //~^ `elif` is not valid syntax
        println!("Bye.");
    }

    // Don't error on valid 'elif' following if block
    let elif = ();
    if true {} elif
}
