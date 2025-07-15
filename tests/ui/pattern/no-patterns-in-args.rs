extern "C" {
    fn f1(mut arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn f2(&arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn f3(arg @ _: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn g1(arg: u8); // OK
    fn g2(_: u8); // OK
// fn g3(u8); // Not yet
}

struct Foo {
    bar: u32
}

type A1 = fn(mut arg: u8); //~ ERROR patterns aren't allowed in function pointer types [E0561]
type A2 = fn(&arg: u8); //~ ERROR patterns aren't allowed in function pointer types [E0561]
type A3 = fn(&_: ()); //~ ERROR patterns aren't allowed in function pointer types [E0561]
type A4 = fn((): ()); //~ ERROR patterns aren't allowed in function pointer types [E0561]
type A5 = fn(Foo { bar }: Foo); //~ ERROR patterns aren't allowed in function pointer types [E0561]

type B1 = fn(arg: u8); // OK
type B2 = fn(_: u8); // OK
type B3 = fn(u8); // OK

fn main() {}
