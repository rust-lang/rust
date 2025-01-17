//@ known-bug: #133063

fn foo(x: !) {
    match x {
        (! | !) if false => {}
        _ => {}
    }
}
