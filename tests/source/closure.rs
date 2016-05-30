// Closures

fn main() {
    let square = ( |i:  i32 | i  *  i );

    let commented = |/* first */ a /*argument*/, /* second*/ b: WithType /* argument*/, /* ignored */ _ |
        (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);

    let block_body = move   |xxxxxxxxxxxxxxxxxxxxxxxxxxxxx,  ref  yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy| {
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxx + yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
        };

    let loooooooooooooong_name = |field| {
             // format comments.
             if field.node.attrs.len() > 0 { field.node.attrs[0].span.lo
             } else {
                 field.span.lo
             }};

    let unblock_me = |trivial| {
                         closure()
                     };

    let empty = |arg|    {};

    let simple = |arg| { /*  comment formatting */ foo(arg) };

    let test = |  | { do_something(); do_something_else(); };

    let arg_test = |big_argument_name, test123| looooooooooooooooooong_function_naaaaaaaaaaaaaaaaame();

    let arg_test = |big_argument_name, test123| {looooooooooooooooooong_function_naaaaaaaaaaaaaaaaame()};

    let simple_closure = move ||   ->  () {};

    let closure = |input: Ty| -> Option<String> {
        foo()
    };

    let closure_with_return_type = |aaaaaaaaaaaaaaaaaaaaaaarg1, aaaaaaaaaaaaaaaaaaaaaaarg2| -> Strong { "sup".to_owned() };

    |arg1, arg2, _, _, arg3, arg4| { let temp = arg4 + arg3;
                                     arg2 * arg1 - temp }
}

fn issue311() {
    let func = |x| println!("{}", x);

    (func)(0.0);
}

fn issue863() {
    let closure = |x| match x {
        0 => true,
        _ => false,
    } == true;
}

fn issue934() {
    let hash: &Fn(&&Block) -> u64 = &|block| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_block(block);
        h.finish()
    };

    let hash: &Fn(&&Block) -> u64 = &|block| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_block(block);
        h.finish();
    };
}

impl<'a, 'tcx: 'a> SpanlessEq<'a, 'tcx> {
    pub fn eq_expr(&self, left: &Expr, right: &Expr) -> bool {
        match (&left.node, &right.node) {
            (&ExprBinary(l_op, ref ll, ref lr), &ExprBinary(r_op, ref rl, ref rr)) => {
                l_op.node == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr) ||
                swap_binop(l_op.node, ll, lr).map_or(false, |(l_op, ll, lr)| l_op == r_op.node && self.eq_expr(ll, rl) && self.eq_expr(lr, rr))
            }
        }
    }
}
