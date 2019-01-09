fn main() {
    1 = 2; //~ ERROR invalid left-hand side expression
    1 += 2; //~ ERROR invalid left-hand side expression
    (1, 2) = (3, 4); //~ ERROR invalid left-hand side expression

    let (a, b) = (1, 2);
    (a, b) = (3, 4); //~ ERROR invalid left-hand side expression

    None = Some(3); //~ ERROR invalid left-hand side expression
}
