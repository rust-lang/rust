fn main() {
    "".chars().fold(|_, _| (), ());
    //~^ ERROR E0277
}
