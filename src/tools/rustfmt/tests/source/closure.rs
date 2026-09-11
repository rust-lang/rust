// rustfmt-normalize_comments: true
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
             if field.node.attrs.len() > 0 { field.node.attrs[0].span.lo()
             } else {
                 field.span.lo()
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
                                     arg2 * arg1 - temp };

    let block_body_with_comment = args.iter()
        .map(|a| {
            // Emitting only dep-info is possible only for final crate type, as
            // as others may emit required metadata for dependent crate types
            if a.starts_with("--emit") && is_final_crate_type && !self.workspace_mode {
                "--emit=dep-info"
            } else { a }
        });

    for<>          || -> () {};
    for<         >|| -> () {};
    for<
>   || -> () {};

for<   'a
   ,'b,
'c  >   |_: &'a (), _: &'b (), _: &'c ()| -> () {};

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

fn foo() {
    lifetimes_iter___map(|lasdfasfd| {
        let hi = if l.bounds.is_empty() {
            l.lifetime.span.hi()
        };
    });
}

fn issue1405() {
    open_raw_fd(fd, b'r')
        .and_then(|file| Capture::new_raw(None, |_, err| unsafe {
            raw::pcap_fopen_offline(file, err)
        }));
}

fn issue1466() {
    let vertex_buffer = frame.scope(|ctx| {
        let buffer =
            ctx.create_host_visible_buffer::<VertexBuffer<Vertex>>(&vertices);
        ctx.create_device_local_buffer(buffer)
    });
}

fn issue470() {
    {{{
        let explicit_arg_decls =
            explicit_arguments.into_iter()
            .enumerate()
            .map(|(index, (ty, pattern))| {
                let lvalue = Lvalue::Arg(index as u32);
                block = this.pattern(block,
                                     argument_extent,
                                     hair::PatternRef::Hair(pattern),
                                     &lvalue);
                ArgDecl { ty: ty }
            });
    }}}
}

// #1509
impl Foo {
    pub fn bar(&self) {
        Some(SomeType {
            push_closure_out_to_100_chars: iter(otherwise_it_works_ok.into_iter().map(|f| {
                Ok(f)
            })),
        })
    }
}

fn issue1329() {
    aaaaaaaaaaaaaaaa.map(|x| {
        x += 1;
        x
    })
        .filter
}

fn issue325() {
    let f = || unsafe { xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx };
}

fn issue1697() {
    Test.func_a(A_VERY_LONG_CONST_VARIABLE_NAME, move |arg1, arg2, arg3, arg4| arg1 + arg2 + arg3 + arg4)
}

fn issue1694() {
    foooooo(|_referencefffffffff: _, _target_reference: _, _oid: _, _target_oid: _| format!("refs/pull/{}/merge", pr_id))
}

fn issue1713() {
    rayon::join(
        || recurse(left, is_less, pred, limit),
        || recurse(right, is_less, Some(pivot), limit),
    );

    rayon::join(
        1,
        || recurse(left, is_less, pred, limit),
        2,
        || recurse(right, is_less, Some(pivot), limit),
    );
}

fn issue2063() {
    |ctx: Ctx<(String, String)>| -> io::Result<Response> {
        Ok(Response::new().with_body(ctx.params.0))
    }
}

fn issue1524() {
    let f = |x| {{{{x}}}};
    let f = |x| {{{x}}};
    let f = |x| {{x}};
    let f = |x| {x};
    let f = |x| x;
}

fn issue2171() {
    foo(|| unsafe {
        if PERIPHERALS {
            loop {}
        } else {
            PERIPHERALS = true;
        }
    })
}

fn issue2207() {
    a.map(|_| unsafe {
        a_very_very_very_very_very_very_very_long_function_name_or_anything_else()
    }.to_string())
}

fn issue2262() {
    result.init(&mut result.slave.borrow_mut(), &mut (result.strategy)()).map_err(|factory| Error {
        factory,
        slave: None,
    })?;
}
