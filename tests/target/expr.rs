// Test expressions

fn foo() -> bool {
    let boxed: Box<i32> = box 5;
    let referenced = &5;

    let very_long_variable_name = (a + first + simple + test);
    let very_long_variable_name = (a + first + simple + test + AAAAAAAAAAAAA +
                                   BBBBBBBBBBBBBBBBB + b + c);

    let is_internalxxxx = self.codemap.span_to_filename(s) ==
                          self.codemap.span_to_filename(m.inner);

    let some_val = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa * bbbb /
                   (bbbbbb - function_call(x, *very_long_pointer, y)) + 1000;

    some_ridiculously_loooooooooooooooooooooong_function(10000 * 30000000000 +
                                                         40000 / 1002200000000 -
                                                         50000 * sqrt(-1),
                                                         trivial_value);
    (((((((((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
             a +
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
             aaaaa)))))))));

    {
        for _ in 0..10 {}
    }

    {
        {
            {
                {}
            }
        }
    }

    if 1 + 2 > 0 {
        let result = 5;
        result
    } else {
        4
    };

    if let Some(x) = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa {
        // Nothing
    }

    if let Some(x) = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}

    if let (some_very_large,
            tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) = 1 + 2 + 3 {
    }

    if let (some_very_large,
            tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) = 1111 +
                                                                                         2222 {}

    if let (some_very_large, tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple) =
           1 + 2 + 3 {
    }

    let test = if true {
        5
    } else {
        3
    };

    if cond() {
        something();
    } else if different_cond() {
        something_else();
    } else {
        // Check subformatting
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    }
}

fn bar() {
    let range = (111111111 + 333333333333333333 + 1111 + 400000000000000000)..(2222 +
                                                                               2333333333333333);

    let another_range = 5..some_func(a, b /* comment */);

    for _ in 1.. {
        call_forever();
    }

    syntactically_correct(loop {
                              sup('?');
                          },
                          if cond {
                              0
                          } else {
                              1
                          });

    let third = ..10;
    let infi_range = ..;
    let foo = 1..;
    let bar = 5;
    let nonsense = (10..0)..(0..10);

    loop {
        if true {
            break
        }
    }

    let x = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
             a);
}

fn baz() {
    unsafe /* {}{}{}{{{{}} */ {
        let foo = 1u32;
    }

    unsafe /* very looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
            * comment */ {
    }

    unsafe /* So this is a very long comment.
            * Multi-line, too.
            * Will it still format correctly? */ {
    }

    unsafe {
        // Regular unsafe block
    }

    unsafe { foo() }

    unsafe {
        foo();
    }
}

// Test some empty blocks.
fn qux() {
    {}
    // FIXME this one could be done better.
    { /* a block with a comment */
    }
    {

    }
    {
        // A block with a comment.
    }
}

fn issue227() {
    {
        let handler = box DocumentProgressHandler::new(addr,
                                                       DocumentProgressTask::DOMContentLoaded);
    }
}

fn issue184(source: &str) {
    for c in source.chars() {
        if index < 'a' {
            continue;
        }
    }
}

fn matches() {
    match 1 {
        1 => 1, // foo
        2 => 2,
        // bar
        3 => 3,
        _ => 0, // baz
    }
}

fn issue339() {
    match a {
        b => {}
        c => {}
        d => {}
        e => {}
        // collapsing here is safe
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff => {}
        // collapsing here exceeds line length
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffg => {
        }
        h => { // comment above block
        }
        i => {} // comment below block
        j => {
            // comment inside block
        }
        j2 => {
            // comments inside...
        } // ... and after
        // TODO uncomment when vertical whitespace is handled better
        // k => {
        //
        //     // comment with WS above
        // }
        // l => {
        //     // comment with ws below
        //
        // }
        m => {}
        n => {}
        o => {}
        p => { // Dont collapse me
        }
        q => {}
        r => {}
        s => 0, // s comment
        // t comment
        t => 1,
        u => 2,
        // TODO uncomment when block-support exists
        // v => {
        // } /* funky block
        //    * comment */
        // final comment
    }
}

fn issue355() {
    match mac {
        a => println!("a", b),
        b => vec!(1, 2),
        c => vec!(3; 4),
        d => {
            println!("a", b)
        }
        e => {
            vec!(1, 2)
        }
        f => {
            vec!(3; 4)
        }
        h => println!("a", b), // h comment
        i => vec!(1, 2), // i comment
        j => vec!(3; 4), // j comment
        // k comment
        k => println!("a", b),
        // l comment
        l => vec!(1, 2),
        // m comment
        m => vec!(3; 4),
        // Rewrite splits macro
        nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn => println!("a",
                                                                                             b),
        // Rewrite splits macro
        oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo => vec!(1,
                                                                                               2),
        // Macro support fails to recognise this macro as splitable
        // We push the whole expr to a new line, TODO split this macro as well
        pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp =>
            vec!(3; 4),
        // q, r and s: Rewrite splits match arm
        qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq =>
            println!("a", b),
        rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr =>
            vec!(1, 2),
        ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss =>
            vec!(3; 4),
        // Funky bracketing styles
        t => println!{"a", b},
        u => vec!{1, 2},
        v => vec!{3; 4},
        w => println!["a", b],
        x => vec![1, 2],
        y => vec![3; 4],
        // Brackets with comments
        tc => println!{"a", b}, // comment
        uc => vec!{1, 2}, // comment
        vc => vec!{3; 4}, // comment
        wc => println!["a", b], // comment
        xc => vec![1, 2], // comment
        yc => vec![3; 4], // comment
    }
}

fn arrays() {
    let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 7, 8, 9, 0, 1, 2, 3, 4,
             5, 6, 7, 8, 9, 0];

    let y = [// comment
             1,
             2, // post comment
             3];

    let z = [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
             yyyyyyyyyyyyyyyyyyyyyyyyyyy,
             zzzzzzzzzzzzzzzzzz,
             q];

    [1 + 3, 4, 5, 6, 7, 7, fncall::<Vec<_>>(3 - 1)]
}

fn returns() {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&
    return;

    return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;
}

fn addrof() {
    &mut (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
          bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);
    &(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +
      bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);
}
