fn main() {
    (0..4)
    .map(|x| x * 2)
    .collect::<Vec<'a, usize, 'b>>()
    //~^ ERROR lifetime parameters must be declared prior to type parameters
}
