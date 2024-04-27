fn main() {
    let const { () } = ();
    //~^ ERROR inline-const in pattern position is experimental [E0658]
}
