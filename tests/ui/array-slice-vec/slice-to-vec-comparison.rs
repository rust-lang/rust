fn main() {
    let a = &[];
    let b: &Vec<u8> = &vec![];
    a > b;
    //~^ ERROR can't compare `[_; 0]` with `Vec<u8>`
}
