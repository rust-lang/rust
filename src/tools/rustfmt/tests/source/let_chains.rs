// rustfmt-edition: 2024

fn main() {
    if let x = x && x {}

    if xxx && let x = x {}

    if aaaaaaaaaaaaaaaaaaaaa &&  aaaaaaaaaaaaaaa && aaaaaaaaa && let Some(x) = xxxxxxxxxxxx && aaaaaaa && let None = aaaaaaaaaa {}

    if aaaaaaaaaaaaaaaaaaaaa &&  aaaaaaaaaaaaaaa && aaaaaaaaa && let Some(x) = xxxxxxxxxxxx && aaaaaaa && let None = aaaaaaaaaa {}

    if let Some(Struct { x:TS(1,2) }) = path::to::<_>(hehe)
        && let [Simple, people] = /* get ready */ create_universe(/* hi */  GreatPowers).initialize_badminton().populate_swamps() &&
        let    everybody    =    (Loops { hi /*hi*/  , ..loopy() }) && summons::triumphantly() { todo!() }

    if let XXXXXXXXX { xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, yyyyyyyyyyyyy, zzzzzzzzzzzzz} = xxxxxxx()
    && let Foo = bar() { todo!() }
}

fn test_single_line_let_chain() {
    // first item in let-chain is an ident
    if a && let Some(b) = foo() {
    }

    // first item in let-chain is a bool literal
    if true && let Some(x) = y {

    }

    // first item in let-chain is a unary ! with an ident
    let unary_not = if !from_hir_call
        && let Some(p) = parent
    {
    };

    // first item in let-chain is a unary * with an ident
    let unary_deref = if *some_deref
        && let Some(p) = parent
    {
    };

    // first item in let-chain is a unary - (neg) with an ident
    let unary_neg = if -some_ident
        && let Some(p) = parent
    {
    };

    // first item in let-chain is a try (?) with an ident
    let try_ = if some_try?
        && let Some(p) = parent
    {
    };

    // first item in let-chain is an ident wrapped in parens
    let in_parens = if (some_ident)
        && let Some(p) = parent
    {
    };

    // first item in let-chain is a ref & with an ident
    let _ref = if &some_ref
        && let Some(p) = parent
    {
    };

    // first item in let-chain is a ref &mut with an ident
    let mut_ref = if &mut some_ref
        && let Some(p) = parent
    {
    };

    // chain unary ref and try
    let chain_of_unary_ref_and_try = if !&*some_ref?
        && let Some(p) = parent {
    };
}

fn test_multi_line_let_chain() {
    // Can only single line the let-chain if the first item is an ident
    if let Some(x) = y && a {

    }

    // More than one let-chain must be formatted on multiple lines
    if let Some(x) = y && let Some(a) = b {

    }

    // The ident isn't long enough so we don't wrap the first let-chain
    if a && let Some(x) = y && let Some(a) = b {

    }

    // The ident is long enough so both let-chains are wrapped
    if aaa && let Some(x) = y && let Some(a) = b {

    }

    // function call
    if a() && let Some(x) = y {

    }

    // cast to a bool
    if 1 as bool && let Some(x) = y {

    }

    // matches! macro call
    if matches!(a, some_type) && let Some(x) = y {

    }

    // block expression returning bool
    if { true } && let Some(x) = y {

    }

    // field access
    if a.x && let Some(x) = y {

    }
}
