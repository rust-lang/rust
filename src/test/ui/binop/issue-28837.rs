struct A;

fn main() {
    let a = A;

    a + a; //~ ERROR cannot add `A` to `A`

    a - a; //~ ERROR cannot subtract `A` from `A`

    a * a; //~ ERROR cannot multiply `A` by `A`

    a / a; //~ ERROR cannot divide `A` by `A`

    a % a; //~ ERROR cannot mod `A` by `A`

    a & a; //~ ERROR no implementation for `A & A`

    a | a; //~ ERROR no implementation for `A | A`

    a << a; //~ ERROR no implementation for `A << A`

    a >> a; //~ ERROR no implementation for `A >> A`

    a == a; //~ ERROR binary operation `==` cannot be applied to type `A`

    a != a; //~ ERROR binary operation `!=` cannot be applied to type `A`

    a < a; //~ ERROR binary operation `<` cannot be applied to type `A`

    a <= a; //~ ERROR binary operation `<=` cannot be applied to type `A`

    a > a; //~ ERROR binary operation `>` cannot be applied to type `A`

    a >= a; //~ ERROR binary operation `>=` cannot be applied to type `A`
}
