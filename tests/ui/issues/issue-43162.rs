fn foo() -> bool {
    //~^ ERROR E0308
    break true; //~ ERROR E0268
}

fn main() {
    break {}; //~ ERROR E0268
}
