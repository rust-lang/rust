fn main() {
    let _ : &(Send,) = &((),);
    //~^ ERROR unsized tuple coercion is not stable enough
}
