fn main() {
    let _ : &(dyn Send,) = &((),);
    //~^ ERROR 2:28: 2:34: mismatched types [E0308]
}
