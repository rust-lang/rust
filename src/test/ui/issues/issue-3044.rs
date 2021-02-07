fn main() {
    let needlesArr: Vec<char> = vec!['a', 'f'];
    needlesArr.iter().fold(|x, y| {
    });
    //~^^ ERROR mismatched types
    //~| ERROR arguments to this function are incorrect
}
