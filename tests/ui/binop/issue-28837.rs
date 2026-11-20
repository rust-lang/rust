struct A;

fn main() {
    let a = A;

    a + a; //~ ERROR: can't add `A`

    a - a; //~ ERROR: can't subtract `A`

    a * a; //~ ERROR: can't multiply `A`

    a / a; //~ ERROR: can't divide `A` by `A`

    a % a; //~ ERROR: can't calculate the remainder of `A` divided by `A`

    a & a; //~ ERROR: no implementation for `A & A`

    a | a; //~ ERROR: no implementation for `A | A`

    a << a; //~ ERROR: no implementation for `A << A`

    a >> a; //~ ERROR: no implementation for `A >> A`

    a == a; //~ ERROR: can't compare `A`

    a != a; //~ ERROR: can't compare `A`

    a < a; //~ ERROR: binary operation `<` can't be applied to type `A`

    a <= a; //~ ERROR: binary operation `<=` can't be applied to type `A`

    a > a; //~ ERROR: binary operation `>` can't be applied to type `A`

    a >= a; //~ ERROR: binary operation `>=` can't be applied to type `A`
}
