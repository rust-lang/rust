fn main() {
    let tup = (0, 1, 2);
    // the case where we show a suggestion
    let _ = tup[0];
    //~^ ERROR cannot index into a value of type

    // the case where we show just a general hint
    let i = 0_usize;
    let _ = tup[i];
    //~^ ERROR cannot index into a value of type
}
