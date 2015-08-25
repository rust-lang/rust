// Test expressions

fn foo() -> bool {
    let boxed: Box<i32> = box   5;
    let referenced = &5 ;

    let very_long_variable_name = ( a +  first +   simple + test   );
    let very_long_variable_name = (a + first + simple + test + AAAAAAAAAAAAA + BBBBBBBBBBBBBBBBB + b + c);

    let is_internalxxxx = self.codemap.span_to_filename(s) == self.codemap.span_to_filename(m.inner);

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

    if cond() {
        something();
    } else  if different_cond() {
        something_else();
    } else {
        // Check subformatting
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    }
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
