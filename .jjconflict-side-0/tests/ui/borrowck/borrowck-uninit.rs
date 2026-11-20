fn foo(x: isize) { println!("{}", x); }

fn main() {
    let x: isize;
    foo(x); //~ ERROR E0381

    // test for #120634
    struct A(u8);
    struct B { d: u8 }
    let (a, );
    let [b, ];
    let A(c);
    let B { d };
    let _: (u8, u8, u8, u8) = (a, b, c, d);
    //~^ ERROR used binding `a`
    //~| ERROR used binding `b`
    //~| ERROR used binding `c`
    //~| ERROR used binding `d`
}
