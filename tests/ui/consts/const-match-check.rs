// revisions: matchck eval1 eval2

#[cfg(matchck)]
const X: i32 = { let 0 = 0; 0 };
//[matchck]~^ ERROR refutable pattern in local binding

#[cfg(matchck)]
static Y: i32 = { let 0 = 0; 0 };
//[matchck]~^ ERROR refutable pattern in local binding

#[cfg(matchck)]
trait Bar {
    const X: i32 = { let 0 = 0; 0 };
    //[matchck]~^ ERROR refutable pattern in local binding
}

#[cfg(matchck)]
impl Bar for () {
    const X: i32 = { let 0 = 0; 0 };
    //[matchck]~^ ERROR refutable pattern in local binding
}

#[cfg(eval1)]
enum Foo {
    A = { let 0 = 0; 0 },
    //[eval1]~^ ERROR refutable pattern in local binding
}

fn main() {
    #[cfg(eval2)]
    let x: [i32; { let 0 = 0; 0 }] = [];
    //[eval2]~^ ERROR refutable pattern in local binding
}
