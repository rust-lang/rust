//@ edition: 2024

trait T {
    fn foo(&self) -> i32;
    fn bar(&self) -> i32;
}

impl<'a> dyn T + 'a {
    fn foo(&self) -> i32 {
        1
    }

    fn bar(&self) {}
}

impl T for i32 {
    fn foo(&self) -> i32 {
        0
    }

    fn bar(&self) -> i32 {
        0
    }
}

trait OtherTrait {
    fn foo(&self) -> i32 {
        i32::MIN
    }
}

impl OtherTrait for dyn T {}

fn main() {
    let x = &0i32;
    assert_eq!(x.foo(), 0);
    assert_eq!(x.bar(), 0);

    let x: &dyn T = &0i32;
    assert_eq!(x.foo(), 1);
    //~^ ERROR multiple applicable items in scope
    assert_eq!(x.bar(), ());
    //~^ ERROR multiple applicable items in scope
    assert_eq!(<dyn T>::foo(x), 1);
    //~^ ERROR multiple applicable items in scope
    assert_eq!(<dyn T as T>::foo(x), 0);
    assert_eq!(<dyn T>::bar(x), ());
    //~^ ERROR multiple applicable items in scope
    assert_eq!(<dyn T as T>::bar(x), 0);
    assert_eq!(<dyn T as OtherTrait>::foo(x), i32::MIN);

    let x: &(dyn T + Send) = &0i32;
    assert_eq!(x.foo(), 0);
}
