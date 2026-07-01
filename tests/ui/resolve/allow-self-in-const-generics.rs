// Allow Self in const generics when Self doesn't depends on generics
trait MyTrait {
    fn foo<const N: i32>();
}

impl MyTrait for i32 {
    fn foo<const N: Self>() {}
}
impl<T> Wrap<T> {
    fn f<const N: Self>() {}
    //~^ ERROR the type of const parameters must not depend on other generic parameters

}

struct Wrap<T>(T);
fn main(){}
