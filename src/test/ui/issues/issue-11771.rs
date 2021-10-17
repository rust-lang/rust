fn main() {
    let x = ();
    1 +
    x //~^ ERROR cannot add
    ;

    let x: () = ();
    1 +
    x //~^ ERROR cannot add
    ;
}
