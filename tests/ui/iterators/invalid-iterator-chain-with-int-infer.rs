fn main() {
    let x = Some(()).iter().map(|()| 1).sum::<f32>();
    //~^ ERROR a value of type `f32` cannot be made by summing an iterator over elements of type `{integer}`
}
