struct S { x: u8, y: u8 }

fn main() {
    let (a, b) = (1, 2);

    (a, b) = (3, 4);
    (a, b) += (3, 4); //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    [a, b] = [3, 4];
    [a, b] += [3, 4]; //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    let s = S { x: 3, y: 4 };

    S { x: a, y: b } = s;
    S { x: a, y: b } += s; //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    S { x: a, ..s } = S { x: 3, y: 4 };
    //~^ ERROR functional record updates are not allowed in destructuring assignments
}
