fn main() {
    loop {
        break.push(1) //~ ERROR no method named `push` found for type `!`
        ;
    }
}
