fn main() {
    1 = 2; //~ ERROR invalid left-hand side of assignment
    1 += 2; //~ ERROR invalid left-hand side of assignment
    (1, 2) = (3, 4); //~ ERROR destructuring assignments are unstable
    //~| ERROR invalid left-hand side of assignment
    //~| ERROR invalid left-hand side of assignment

    let (a, b) = (1, 2);
    (a, b) = (3, 4); //~ ERROR destructuring assignments are unstable

    None = Some(3); //~ ERROR invalid left-hand side of assignment
}
