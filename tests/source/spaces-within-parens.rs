// rustfmt-spaces_within_parens: true

enum E {
    A(u32),
    B(u32, u32),
    C(u32, u32, u32),
    D(),
}

struct TupleStruct0();
struct TupleStruct1(u32);
struct TupleStruct2(u32, u32);

fn fooEmpty() {}

fn foo(e: E, _: u32) -> (u32, u32) {

    // Tuples
    let t1 = ();
    let t2 = (1,);
    let t3 = (1, 2);

    let ts0 = TupleStruct0();
    let ts1 = TupleStruct1(1);
    let ts2 = TupleStruct2(1, 2);

    // Tuple pattern
    let (a,b,c) = (1,2,3);

    // Expressions
    let x = (1 + 2) * (3);

    // Function call
    fooEmpty();
    foo(1, 2);

    // Pattern matching
    match e {
        A(_) => (),
        B(_, _) => (),
        C(..) => (),
        D => (),
    }

    (1,2)
}
