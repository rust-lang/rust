//@ dont-require-annotations: HELP

// FieldExpression, TupleIndexingExpression
fn field_access() {
    match 0 {
        x => (),
        x.y => (), //~ error: expected a pattern, found an expression
        x.0 => (), //~ error: expected a pattern, found an expression
        x._0 => (), //~ error: expected a pattern, found an expression
        x.0.1 => (), //~ error: expected a pattern, found an expression
        x.4.y.17.__z => (), //~ error: expected a pattern, found an expression
    }

    { let x.0e0; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `.`
    { let x.-0.0; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `.`
    { let x.-0; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `.`

    { let x.0u32; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `.`
    { let x.0.0_f64; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `.`
}

// IndexExpression, ArrayExpression
fn array_indexing() {
    match 0 {
        x[0] => (), //~ error: expected a pattern, found an expression
        x[..] => (), //~ error: expected a pattern, found an expression
    }

    { let x[0, 1, 2]; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `[`
    { let x[0; 20]; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `[`
    { let x[]; } //~ error: expected one of `:`, `;`, `=`, `@`, or `|`, found `[`
    { let (x[]); } //~ error: expected one of `)`, `,`, `@`, `if`, or `|`, found `[`
    //~^ HELP missing `,`
}

// MethodCallExpression, CallExpression, ErrorPropagationExpression
fn method_call() {
    match 0 {
        x.f() => (), //~ error: expected a pattern, found an expression
        x._f() => (), //~ error: expected a pattern, found an expression
        x? => (), //~ error: expected a pattern, found an expression
        ().f() => (), //~ error: expected a pattern, found an expression
        (0, x)?.f() => (), //~ error: expected a pattern, found an expression
        x.f().g() => (), //~ error: expected a pattern, found an expression
        0.f()?.g()?? => (), //~ error: expected a pattern, found an expression
    }
}

// TypeCastExpression
fn type_cast() {
    match 0 {
        x as usize => (), //~ error: expected a pattern, found an expression
        0 as usize => (), //~ error: expected a pattern, found an expression
        x.f().0.4 as f32 => (), //~ error: expected a pattern, found an expression
    }
}

// ArithmeticOrLogicalExpression, also check if parentheses are added as needed
fn operator() {
    match 0 {
        1 + 1 => (), //~ error: expected a pattern, found an expression
        (1 + 2) * 3 => (),
        //~^ error: expected a pattern, found an expression
        //~| error: expected a pattern, found an expression
        x.0 > 2 => (), //~ error: expected a pattern, found an expression
        x.0 == 2 => (), //~ error: expected a pattern, found an expression
    }

    // preexisting match arm guard
    match (0, 0) {
        (x, y.0 > 2) if x != 0 => (), //~ error: expected a pattern, found an expression
        (x, y.0 > 2) if x != 0 || x != 1 => (), //~ error: expected a pattern, found an expression
    }
}

const _: u32 = match 12 {
    1 + 2 * PI.cos() => 2, //~ error: expected a pattern, found an expression
    _ => 0,
};

fn main() {
    match u8::MAX {
        u8::MAX.abs() => (),
        //~^ error: expected a pattern, found an expression
        x.sqrt() @ .. => (),
        //~^ error: expected a pattern, found an expression
        //~| error: left-hand side of `@` must be a binding
        z @ w @ v.u() => (),
        //~^ error: expected a pattern, found an expression
        y.ilog(3) => (),
        //~^ error: expected a pattern, found an expression
        n + 1 => (),
        //~^ error: expected a pattern, found an expression
        ("".f() + 14 * 8) => (),
        //~^ error: expected a pattern, found an expression
        0 | ((1) | 2) | 3 => (),
        f?() => (),
        //~^ error: expected a pattern, found an expression
        (_ + 1) => (),
        //~^ error: expected one of `)`, `,`, `if`, or `|`, found `+`
    }

    let 1 + 1 = 2;
    //~^ error: expected a pattern, found an expression

    let b = matches!(x, (x * x | x.f()) | x[0]);
    //~^ error: expected one of `)`, `,`, `@`, `if`, or `|`, found `*`
}
