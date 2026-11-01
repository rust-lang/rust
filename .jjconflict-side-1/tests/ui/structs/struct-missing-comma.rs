// Issue #50636
//@ run-rustfix

pub struct S {
    pub foo: u32 //~ ERROR expected `,`, or `}`, found keyword `pub`
    //     ~^ HELP try adding a comma: ','
    pub bar: u32
}

fn main() {
    let _ = S { foo: 5, bar: 6 };
}
