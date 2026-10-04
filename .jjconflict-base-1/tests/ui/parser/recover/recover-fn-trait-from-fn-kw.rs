fn foo(_: impl fn() -> i32) {}
//~^ ERROR expected identifier, found keyword `fn`

fn foo2<T: fn(i32)>(_: T) {}
//~^ ERROR expected identifier, found keyword `fn`

fn main() {
    foo(|| ());
    //~^ ERROR mismatched types
    foo2(|_: ()| {});
    //~^ ERROR type mismatch in closure arguments
}
