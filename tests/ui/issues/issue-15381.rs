fn main() {
    let values: Vec<u8> = vec![1,2,3,4,5,6,7,8];

    for &[x,y,z] in values.chunks(3).filter(|&xs| xs.len() == 3) {
        //~^ ERROR refutable pattern in `for` loop binding
        //~| patterns `&[]`, `&[_]`, `&[_, _]` and 1 more not covered
        println!("y={}", y);
    }
}
