struct A;

fn main() {
    let a = A;

    a + a; //~ ERROR binary operation `+` cannot be applied to type `A`

    a - a; //~ ERROR binary operation `-` cannot be applied to type `A`

    a * a; //~ ERROR binary operation `*` cannot be applied to type `A`

    a / a; //~ ERROR binary operation `/` cannot be applied to type `A`

    a % a; //~ ERROR binary operation `%` cannot be applied to type `A`

    a & a; //~ ERROR binary operation `&` cannot be applied to type `A`

    a | a; //~ ERROR binary operation `|` cannot be applied to type `A`

    a << a; //~ ERROR binary operation `<<` cannot be applied to type `A`

    a >> a; //~ ERROR binary operation `>>` cannot be applied to type `A`

    a == a; //~ ERROR binary operation `==` cannot be applied to type `A`

    a != a; //~ ERROR binary operation `!=` cannot be applied to type `A`

    a < a; //~ ERROR binary operation `<` cannot be applied to type `A`

    a <= a; //~ ERROR binary operation `<=` cannot be applied to type `A`

    a > a; //~ ERROR binary operation `>` cannot be applied to type `A`

    a >= a; //~ ERROR binary operation `>=` cannot be applied to type `A`
}
