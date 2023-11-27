// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

struct MyType;
trait MyTrait {
}

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

impl<T: Copy> MyTrait for T {}
//~^ NOTE first implementation here
impl MyTrait for Box<<(MyType,) as Mirror>::Assoc> {}
//~^ ERROR conflicting implementations of trait `MyTrait` for type `Box<(MyType,)>`
//~| NOTE conflicting implementation for `Box<(MyType,)>
//~| NOTE upstream crates may add a new impl of trait `std::marker::Copy` for type `std::boxed::Box<(MyType,)>` in future versions

fn main() {}
