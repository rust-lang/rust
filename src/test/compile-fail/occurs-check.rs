fn main() {
    let f; //! ERROR this local variable has a type of infinite size
    f = @f;
}
