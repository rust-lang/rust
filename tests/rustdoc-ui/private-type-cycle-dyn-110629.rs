// https://github.com/rust-lang/rust/issues/110629

type Bar<'a, 'b> = Box<dyn PartialEq<Bar<'a, 'b>>>;
//~^ ERROR cycle detected when expanding type alias

fn bar<'a, 'b>(i: &'a i32) -> Bar<'a, 'b> {
    Box::new(i)
}

fn main() {
    let meh = 42;
    let muh = 42;
    assert!(bar(&meh) == bar(&muh));
}
