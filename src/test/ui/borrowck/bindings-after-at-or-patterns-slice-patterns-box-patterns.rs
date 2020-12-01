// Tests using a combination of pattern features has the expected borrow checking behavior
#![feature(bindings_after_at)]
#![feature(or_patterns)]
#![feature(box_patterns)]

enum Test {
    Foo,
    Bar,
    _Baz,
}

// bindings_after_at + slice_patterns

fn bindings_after_at_slice_patterns_move_binding(x: [String; 4]) {
    match x {
        a @ [.., _] => (),
        _ => (),
    };

    &x;
    //~^ ERROR borrow of moved value
}

fn bindings_after_at_slice_patterns_borrows_binding_mut(mut x: [String; 4]) {
    let r = match x {
        ref mut foo @ [.., _] => Some(foo),
        _ => None,
    };

    &x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_slice_patterns_borrows_slice_mut1(mut x: [String; 4]) {
    let r = match x {
        ref foo @ [.., ref mut bar] => (),
        //~^ ERROR cannot borrow
        _ => (),
    };

    drop(r);
}

fn bindings_after_at_slice_patterns_borrows_slice_mut2(mut x: [String; 4]) {
    let r = match x {
        [ref foo @ .., ref bar] => Some(foo),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_slice_patterns_borrows_both(mut x: [String; 4]) {
    let r = match x {
        ref foo @ [.., ref bar] => Some(foo),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

// bindings_after_at + or_patterns

fn bindings_after_at_or_patterns_move(x: Option<Test>) {
    match x {
        foo @ Some(Test::Foo | Test::Bar) => (),
        _ => (),
    }

    &x;
    //~^ ERROR borrow of moved value
}

fn bindings_after_at_or_patterns_borrows(mut x: Option<Test>) {
    let r = match x {
        ref foo @ Some(Test::Foo | Test::Bar) => Some(foo),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_or_patterns_borrows_mut(mut x: Option<Test>) {
    let r = match x {
        ref mut foo @ Some(Test::Foo | Test::Bar) => Some(foo),
        _ => None,
    };

    &x;
    //~^ ERROR cannot borrow

    drop(r);
}

// bindings_after_at + box_patterns

fn bindings_after_at_box_patterns_borrows_both(mut x: Option<Box<String>>) {
    let r = match x {
        ref foo @ Some(box ref s) => Some(foo),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_box_patterns_borrows_mut(mut x: Option<Box<String>>) {
    match x {
        ref foo @ Some(box ref mut s) => (),
        //~^ ERROR cannot borrow
        _ => (),
    };
}

// bindings_after_at + slice_patterns + or_patterns

fn bindings_after_at_slice_patterns_or_patterns_moves(x: [Option<Test>; 4]) {
    match x {
        a @ [.., Some(Test::Foo | Test::Bar)] => (),
        _ => (),
    };

    &x;
    //~^ ERROR borrow of moved value
}

fn bindings_after_at_slice_patterns_or_patterns_borrows_binding(mut x: [Option<Test>; 4]) {
    let r = match x {
        ref a @ [ref b @ .., Some(Test::Foo | Test::Bar)] => Some(a),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_slice_patterns_or_patterns_borrows_slice(mut x: [Option<Test>; 4]) {
    let r = match x {
        ref a @ [ref b @ .., Some(Test::Foo | Test::Bar)] => Some(b),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

// bindings_after_at + slice_patterns + box_patterns

fn bindings_after_at_slice_patterns_box_patterns_borrows(mut x: [Option<Box<String>>; 4]) {
    let r = match x {
        [_, ref a @ Some(box ref b), ..] => Some(a),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

// bindings_after_at + slice_patterns + or_patterns + box_patterns

fn bindings_after_at_slice_patterns_or_patterns_box_patterns_borrows(
    mut x: [Option<Box<Test>>; 4]
) {
    let r = match x {
        [_, ref a @ Some(box Test::Foo | box Test::Bar), ..] => Some(a),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_slice_patterns_or_patterns_box_patterns_borrows_mut(
    mut x: [Option<Box<Test>>; 4]
) {
    let r = match x {
        [_, ref mut a @ Some(box Test::Foo | box Test::Bar), ..] => Some(a),
        _ => None,
    };

    &x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn bindings_after_at_slice_patterns_or_patterns_box_patterns_borrows_binding(
    mut x: [Option<Box<Test>>; 4]
) {
    let r = match x {
        ref a @ [_, ref b @ Some(box Test::Foo | box Test::Bar), ..] => Some(a),
        _ => None,
    };

    &mut x;
    //~^ ERROR cannot borrow

    drop(r);
}

fn main() {}
