struct S { x: u8, y: u8 }

fn main() {
    let (a, b) = (1, 2);

    (a, b) = (3, 4); //~ ERROR destructuring assignments are unstable
    (a, b) += (3, 4); //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    [a, b] = [3, 4]; //~ ERROR invalid left-hand side of assignment
    [a, b] += [3, 4]; //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    let s = S { x: 3, y: 4 };

    S { x: a, y: b } = s; //~ ERROR invalid left-hand side of assignment
    S { x: a, y: b } += s; //~ ERROR invalid left-hand side of assignment
    //~| ERROR binary assignment operation `+=` cannot be applied

    S { x: a, ..s } = S { x: 3, y: 4 };
    //~^ ERROR invalid left-hand side of assignment

    let c = 3;

    ((a, b), c) = ((3, 4), 5); //~ ERROR destructuring assignments are unstable
}
