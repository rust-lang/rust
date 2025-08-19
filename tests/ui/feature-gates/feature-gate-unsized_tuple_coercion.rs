fn main() {
    let _ : &(dyn Send,) = &((),);
    //~^ ERROR mismatched types [E0308]
}
