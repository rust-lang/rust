// Closures

fn main() {
    let square = (|i: i32| i * i);

    let commented = |// first
                     a, // argument
                     // second
                     b: WithType, // argument
                     // ignored
                     _| {
        (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)
    };

    let block_body = move |xxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
                           ref yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy| {
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxx + yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
    };

    let loooooooooooooong_name = |field| {
             // TODO(#27): format comments.
        if field.node.attrs.len() > 0 {
            field.node.attrs[0].span.lo
        } else {
            field.span.lo
        }
    };

    let block_me = |field| {
        if true_story() {
            1
        } else {
            2
        }
    };

    let unblock_me = |trivial| closure();

    let empty = |arg| {};

    let simple = |arg| { /* TODO(#27): comment formatting */
        foo(arg)
    };

    let test = || {
        do_something();
        do_something_else();
    };

    let arg_test = |big_argument_name, test123| {
        looooooooooooooooooong_function_naaaaaaaaaaaaaaaaame()
    };

    let arg_test = |big_argument_name, test123| {
        looooooooooooooooooong_function_naaaaaaaaaaaaaaaaame()
    };

    let simple_closure = move || -> () {};

    let closure = |input: Ty| -> Option<String> { foo() };

    let closure_with_return_type = |aaaaaaaaaaaaaaaaaaaaaaarg1,
                                    aaaaaaaaaaaaaaaaaaaaaaarg2|
                                    -> Strong {
        "sup".to_owned()
    };

    |arg1, arg2, _, _, arg3, arg4| {
        let temp = arg4 + arg3;
        arg2 * arg1 - temp
    }
}

fn issue311() {
    let func = |x| println!("{}", x);

    (func)(0.0);
}
