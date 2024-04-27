fn main() {
    let x = ();
    1 +
    x //~^ ERROR E0277
    ;

    let x: () = ();
    1 +
    x //~^ ERROR E0277
    ;
}
