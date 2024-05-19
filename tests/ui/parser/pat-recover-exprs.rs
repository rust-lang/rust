fn main() {
    match u8::MAX {
        u8::MAX.abs() => (),
        //~^ error: expected a pattern, found a method call
        x.sqrt() @ .. => (),
        //~^ error: expected a pattern, found a method call
        //~| error: left-hand side of `@` must be a binding
        z @ w @ v.u() => (),
        //~^ error: expected a pattern, found a method call
        y.ilog(3) => (),
        //~^ error: expected a pattern, found a method call
        n + 1 => (),
        //~^ error: expected a pattern, found an expression
        ("".f() + 14 * 8) => (),
        //~^ error: expected a pattern, found an expression
        0 | ((1) | 2) | 3 => (),
        f?() => (),
        //~^ error: expected a pattern, found an expression
        (_ + 1) => (),
        //~^ error: expected one of `)`, `,`, or `|`, found `+`
    }

    let 1 + 1 = 2;
    //~^ error: expected a pattern, found an expression

    let b = matches!(x, (x * x | x.f()) | x[0]);
    //~^ error: expected one of `)`, `,`, `@`, or `|`, found `*`
}
