fn main() {
    1 = 2; //~ ERROR invalid left-hand side of assignment
    1 += 2; //~ ERROR invalid left-hand side of assignment
    (1, 2) = (3, 4); //~ ERROR invalid left-hand side of assignment

    let (a, b) = (1, 2);
    (a, b) = (3, 4); //~ ERROR invalid left-hand side of assignment

    None = Some(3); //~ ERROR invalid left-hand side of assignment
}
