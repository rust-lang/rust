struct Irrefutable(i32);

fn main() {
    let irr = Irrefutable(0);
    while let Irrefutable(x) = irr { //~ ERROR E0165
                                     //~| irrefutable pattern
        // ...
    }
}
