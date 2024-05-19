fn main () {
    loop {
        break 'a;
        //~^ ERROR E0426
    }
}
