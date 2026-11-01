// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true
// Test expressions

fn foo() -> bool {
    let referenced = &5 ;

    let very_long_variable_name = ( a +  first +   simple + test   );
    let very_long_variable_name = (a + first + simple + test + AAAAAAAAAAAAA + BBBBBBBBBBBBBBBBB + b + c);

    let is_internalxxxx = self.source_map.span_to_filename(s) == self.source_map.span_to_filename(m.inner);

    let some_val = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa * bbbb / (bbbbbb -
        function_call(x, *very_long_pointer, y))
    + 1000  ;

some_ridiculously_loooooooooooooooooooooong_function(10000 * 30000000000 + 40000 / 1002200000000
                                                     - 50000 * sqrt(-1),
                                                     trivial_value);
    (((((((((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + a +
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaa)))))))))   ;

    { for _ in 0..10 {} }

    {{{{}}}}

     if  1  + 2 > 0  { let result = 5; result } else { 4};

    if  let   Some(x)  =  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa {
        // Nothing
    }

    if  let   Some(x)  =  (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}

    if let (some_very_large,
            tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) = 1
        + 2 + 3 {
    }

    if let (some_very_large,
            tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) = 1111 + 2222 {}

    if let (some_very_large, tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) = 1
 + 2 + 3 {
    }

    if let ast::ItemKind::Trait(_, unsafety, ref generics, ref type_param_bounds, ref trait_items) = item.node
    {
        // nothing
    }

    let test = if true { 5 } else { 3 };

    if cond() {
        something();
    } else  if different_cond() {
        something_else();
    } else {
        // Check subformatting
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    }

    // #2884
    let _ = [0; {struct Foo; impl Foo {const fn get(&self) -> usize {5}}; Foo.get()}];
}

fn bar() {
    let range = (   111111111 + 333333333333333333 + 1111 +   400000000000000000) .. (2222 +  2333333333333333);

    let another_range = 5..some_func( a , b /* comment */);

    for _  in  1 ..{ call_forever(); }

    syntactically_correct(loop { sup( '?'); }, if cond { 0 } else { 1 });

    let third = ..10;
    let infi_range = ..   ;
    let foo = 1..;
    let bar = 5 ;
    let nonsense = (10 .. 0)..(0..10);

    loop{if true {break}}

    let x = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
             a);
}

fn baz() {
    unsafe    /*    {}{}{}{{{{}}   */   {
        let foo = 1u32;
    }

    unsafe /* very looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong comment */ {}

    unsafe // So this is a very long comment.
           // Multi-line, too.
           // Will it still format correctly?
    {
    }

    unsafe {
        // Regular unsafe block
    }

    unsafe {
        foo()
    }

    unsafe {
        foo();
    }

    // #2289
    let identifier_0 = unsafe { this_is_58_chars_long_and_line_is_93_chars_long_xxxxxxxxxx };
    let identifier_1 = unsafe { this_is_59_chars_long_and_line_is_94_chars_long_xxxxxxxxxxx };
    let identifier_2 = unsafe { this_is_65_chars_long_and_line_is_100_chars_long_xxxxxxxxxxxxxxxx };
    let identifier_3 = unsafe { this_is_66_chars_long_and_line_is_101_chars_long_xxxxxxxxxxxxxxxxx };
}

// Test some empty blocks.
fn qux() {
    {}
    // FIXME this one could be done better.
    { /* a block with a comment */ }
    {

    }
    {
        // A block with a comment.
    }
}

fn issue184(source: &str) {
    for c in source.chars() {
        if index < 'a' {
            continue;
        }
    }
}

fn arrays() {
    let x = [0,
         1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         0,
         1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         0,
         7,
         8,
         9,
         0,
         1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         0];

    let y = [/* comment */ 1, 2 /* post comment */, 3];

    let xy =    [ strukt  { test123: value_one_two_three_four, turbo: coolio(), } , /* comment  */   1 ];

        let a =WeightedChoice::new(&mut [Weighted {
            weightweight: x,
            item: 0,
        },
                                  Weighted {
            weightweight: 1,
            item: 1,
        },
                                  Weighted {
            weightweight: x,
            item: 2,
        },
                                  Weighted {
            weightweight: 1,
            item: 3,
        }]);

    let z = [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, yyyyyyyyyyyyyyyyyyyyyyyyyyy, zzzzzzzzzzzzzzzzzz, q];

    [ 1 +   3, 4 ,  5, 6, 7, 7, fncall::<Vec<_>>(3-1)]
}

fn returns() {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa && return;

    return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;
}

fn addrof() {
    &    mut(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);
    &    (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);

    // raw reference operator
    & raw   const  a;
    & raw   mut    b;
}

fn casts() {
    fn unpack(packed: u32) -> [u16; 2] {
        [
            (packed >> 16) as u16,
            (packed >>  0) as u16,
        ]
    }

    let some_trait_xxx = xxxxxxxxxxx + xxxxxxxxxxxxx
        as SomeTraitXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX;
    let slightly_longer_trait = yyyyyyyyy + yyyyyyyyyyy as SomeTraitYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY;
}

fn indices() {
    let x = (aaaaaaaaaaaaaaaaaaaaaaaaaaaa+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb+cccccccccccccccc) [ x + y + z ];
    let y = (aaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb + cccccccccccccccc)[ xxxxx + yyyyy + zzzzz ];
    let z = xxxxxxxxxx.x().y().zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz()[aaaaa];
    let z = xxxxxxxxxx.x().y().zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz()[aaaaa];
}

fn repeats() {
    let x = [aaaaaaaaaaaaaaaaaaaaaaaaaaaa+bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb+cccccccccccccccc; x + y + z ];
    let y = [aaaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb + cccccccccccccccc; xxxxx + yyyyy + zzzzz ];
}

fn blocks() {
    if 1 + 1 == 2 {
        println!("yay arithmetix!");
    };
}

fn issue767() {
    if false {
        if false {
        } else {
            // A let binding here seems necessary to trigger it.
            let _ = ();
        }
    } else if let false = false {
    }
}

fn ranges() {
    let x = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa .. bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
    let y = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ..= bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
    let z = ..= x ;

    // #1766
    let x = [0. ..10.0];
    let x = [0. ..=10.0];

    a ..= b

    // the expr below won't compile because inclusive ranges need a defined end
    // let a = 0 ..= ;
}

fn if_else() {
  let exact = diff /
            (if size == 0 {
    1
} else {
    size
});

            let cx = tp1.x +
                 any * radius *
                 if anticlockwise {
            1.0
      } else {
           -1.0
       };
}

fn complex_if_else() {
    if let Some(x) = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {
    } else if let Some(x) = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {
        ha();
    } else if xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx + xxxxxxxx {
        yo();
    } else if let Some(x) = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {
        ha();
    } else if xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx + xxxxxxxxx {
        yo();
    }
}

fn issue1106() {
    {
    if let hir::ItemEnum(ref enum_def, ref generics) = self.ast_map.expect_item(enum_node_id).node {
    }
    }

    for entry in
    WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| exclusions.filter_entry(entry)) {
    }
}

fn issue1570() {
    a_very_long_function_name({some_func(1, {1})})
}

fn issue1714() {
    v = &mut {v}[mid..];
    let (left, right) = {v}.split_at_mut(mid);
}

// Multi-lined index should be put on the next line if it fits in one line.
fn issue1749() {
    {
        {
            {
                if self.shape[(r as f32 + self.x_offset) as usize][(c as f32 + self.y_offset) as usize] != 0 {
                    // hello
                }
            }
        }
    }
}

// #1172
fn newlines_between_list_like_expr() {
    foo(
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,

        yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,

        zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,
    );

    vec![
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,

        yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,

        zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,
    ];

    match x {
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx |

        yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy |

        zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz => foo(a, b, c),
        _ => bar(),
    };
}

fn issue2178() {
    Ok(result.iter().map(|item| ls_util::rls_to_location(item)).collect())
}

// #2493
impl Foo {
fn bar(&self) {
    {
        let x = match () {
            () => {
                let i;
                i == self.install_config.storage.experimental_compressed_block_size as usize
            }
        };
    }
}
}

fn dots() {
    .. .. ..; // (.. (.. (..)))
    ..= ..= ..;
    (..) .. ..; // ((..) .. (..))
}

// #2676
// A function call with a large single argument.
fn foo() {
    let my_var =
        Mutex::new(RpcClientType::connect(server_iddd).chain_err(|| "Unable to create RPC client")?);
}

// #2704
// Method call with prefix and suffix.
fn issue2704() {
    // We should not combine the callee with a multi-lined method call.
    let requires = requires.set(&requires0
                                .concat(&requires1)
                                .concat(&requires2)
                                .distinct_total());
    let requires = requires.set(requires0
                                .concat(&requires1)
                                .concat(&requires2)
                                .distinct_total() as u32);
    let requires = requires.set(requires0
                                .concat(&requires1)
                                .concat(&requires2)
                                .distinct_total()?);
    let requires = requires.set(!requires0
                                .concat(&requires1)
                                .concat(&requires2)
                                .distinct_total());
    // We should combine a small callee with an argument.
    bar(vec![22]
        .into_iter()
        .map(|x| x * 2)
        .filter(|_| true)
        .collect());
    // But we should not combine a long callee with an argument.
    barrrr(vec![22]
        .into_iter()
        .map(|x| x * 2)
        .filter(|_| true)
        .collect());
}

// #2782
fn issue2782() {
    {let f={let f={{match f{F(f,_)=>{{loop{let f={match f{F(f,_)=>{{match f{F(f,_)=>{{loop{let f={let f={match f{'-'=>F(f,()),}};};}}}}}}}};}}}}}};};}
}

fn issue_2802() {
    function_to_fill_this_line(some_arg, some_arg, some_arg)
        * a_very_specific_length(specific_length_arg) * very_specific_length(Foo {
        a: some_much_much_longer_value,
    }) * some_value
}

fn issue_3003() {
    let mut path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "support",
        "dejavu-fonts-ttf-2.37",
        "ttf",
    ]
        .iter()
        .collect();
}

fn issue3226() {
    {
        {
            {
                return Err(ErrorKind::ManagementInterfaceError("Server exited unexpectedly").into())
            }
        }
    }
    {
        {
            {
                break Err(ErrorKind::ManagementInterfaceError("Server exited unexpectedlyy").into())
            }
        }
    }
}

// #3457
fn issue3457() {
    {
        {
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            {
                                                {
                                                    {
                                                        {
                                                            {
                                                                {
                                                                    {
                                                                        {
                                                                            {
                                                                                {
                                                                                    {
                                                                                        {
                                                                                            {
                                                                                                {
                                                                                                    println!("Test");
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// #3498
static REPRO: &[usize] = &[#[cfg(feature = "zero")]
                           0];

fn overflow_with_attr() {
    foo(#[cfg(feature = "zero")]
        0);
    foobar(#[cfg(feature = "zero")]
           0);
    foobar(x, y, #[cfg(feature = "zero")]
           {});
}


// https://github.com/rust-lang/rustfmt/issues/3765
fn foo() {
    async {
        // Do
        // some
        // work
    }
    .await;

    async {
        // Do
        // some
        // work
    }
        .await;
}

fn underscore() {
        _= 1;
        _;
        [  _,a,_  ] = [1, 2, 3];
    (a,   _) = (8, 9);
    TupleStruct(  _, a) = TupleStruct(2, 2);

    let _  : usize = foo(_, _);
}
