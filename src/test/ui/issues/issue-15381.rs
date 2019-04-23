fn main() {
    let values: Vec<u8> = vec![1,2,3,4,5,6,7,8];

    for &[x,y,z] in values.chunks(3).filter(|&xs| xs.len() == 3) {
        //~^ ERROR refutable pattern in `for` loop binding: `&[]` not covered
        println!("y={}", y);
        //~^ WARN borrow of possibly uninitialized variable: `y`
        //~| WARN this error has been downgraded to a warning for backwards compatibility
        //~| WARN this represents potential undefined behavior in your code and this warning will
    }
}
