// run-rustfix
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let vr = v.iter().filter(|x| {
        *x % 2 == 0
        //~^ ERROR cannot calculate the remainder of `&&{integer}` divided by `{integer}`
    });
    println!("{:?}", vr);
}
