fn main() {
    let needlesArr: Vec<char> = vec!['a', 'f'];
    needlesArr.iter().fold(|x, y| {
        //~^ ERROR this method takes 2 arguments but 1 argument was supplied
    });
}
