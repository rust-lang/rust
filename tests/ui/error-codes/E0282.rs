fn main() {
    let x = "hello".chars().rev().collect();
    //~^ ERROR E0282
    //~| ERROR E0283
}
