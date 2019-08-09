fn main() {
    let _ : &(dyn Send,) = &((),);
    //~^ ERROR unsized tuple coercion is not stable enough
}
