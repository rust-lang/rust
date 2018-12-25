// Test a "pass-through" object-lifetime-default that produces errors.

#![allow(dead_code)]

trait SomeTrait {
    fn dummy(&self) { }
}

struct MyBox<T:?Sized> {
    r: Box<T>
}

fn deref<T>(ss: &T) -> T {
    // produces the type of a deref without worrying about whether a
    // move out would actually be legal
    loop { }
}

fn load0(ss: &MyBox<SomeTrait>) -> MyBox<SomeTrait> {
    deref(ss)
}

fn load1<'a,'b>(a: &'a MyBox<SomeTrait>,
                b: &'b MyBox<SomeTrait>)
                -> &'b MyBox<SomeTrait>
{
    a //~ ERROR lifetime mismatch
}

fn load2<'a>(ss: &MyBox<SomeTrait+'a>) -> MyBox<SomeTrait+'a> {
    load0(ss) //~ ERROR mismatched types
}

fn main() {
}
