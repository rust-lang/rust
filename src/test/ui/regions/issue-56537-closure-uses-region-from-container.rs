// This is a collection of examples where a function's formal
// parameter has an explicit lifetime and a closure within that
// function returns that formal parameter. The closure's return type,
// to be correctly inferred, needs to include the lifetime introduced
// by the function.
//
// This works today, which precludes changing things so that closures
// follow the same lifetime-elision rules used elsehwere. See
// rust-lang/rust#56537

// build-pass (FIXME(62277): could be check-pass?)

fn willy_no_annot<'w>(p: &'w str, q: &str) -> &'w str {
    let free_dumb = |_x| { p }; // no type annotation at all
    let hello = format!("Hello");
    free_dumb(&hello)
}

fn willy_ret_type_annot<'w>(p: &'w str, q: &str) -> &'w str {
    let free_dumb = |_x| -> &str { p }; // type annotation on the return type
    let hello = format!("Hello");
    free_dumb(&hello)
}

fn willy_ret_region_annot<'w>(p: &'w str, q: &str) -> &'w str {
    let free_dumb = |_x| -> &'w str { p }; // type+region annotation on return type
    let hello = format!("Hello");
    free_dumb(&hello)
}

fn willy_arg_type_ret_type_annot<'w>(p: &'w str, q: &str) -> &'w str {
    let free_dumb = |_x: &str| -> &str { p }; // type annotation on arg and return types
    let hello = format!("Hello");
    free_dumb(&hello)
}

fn willy_arg_type_ret_region_annot<'w>(p: &'w str, q: &str) -> &'w str {
    let free_dumb = |_x: &str| -> &'w str { p }; // fully annotated
    let hello = format!("Hello");
    free_dumb(&hello)
}

fn main() {
    let world = format!("World");
    let w1: &str = {
        let hello = format!("He11o");
        willy_no_annot(&world, &hello)
    };
    let w2: &str = {
        let hello = format!("He22o");
        willy_ret_type_annot(&world, &hello)
    };
    let w3: &str = {
        let hello = format!("He33o");
        willy_ret_region_annot(&world, &hello)
    };
    let w4: &str = {
        let hello = format!("He44o");
        willy_arg_type_ret_type_annot(&world, &hello)
    };
    let w5: &str = {
        let hello = format!("He55o");
        willy_arg_type_ret_region_annot(&world, &hello)
    };
    assert_eq!((w1, w2, w3, w4, w5),
               ("World","World","World","World","World"));
}
