// Issue #50636

struct S {
    foo: u32 //~ expected `,`, or `}`, found `bar`
    //     ~^ HELP try adding a comma: ','
    bar: u32
}

fn main() {
    let s = S { foo: 5, bar: 6 };
}
