fn main() {
    let f; //~ ERROR cyclic type of infinite size
    f = @f;
}
