mod a {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod b {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod c {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod d {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod e {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod f {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod g {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod h {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod i {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod j {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod k {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod l {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod m {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod n {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod o {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod p {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod q {
    struct Foo {}
    impl Foo {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a Foo) {
        x.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod r {
    struct Foo {}
    impl Foo {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a Foo) {
        x.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod s {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}

    trait Trait { fn hello(&self) {} }

    impl Trait for dyn Foo {
        fn hello(&self) {}
    }
    fn convert<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod t {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {}
    trait MyTrait where Self: 'static {
        fn use_self(&self) -> &() where Self: 'static { panic!() }
    }
    trait Irrelevant {
        fn use_self(&self) -> &() { panic!() }
    }

    impl MyTrait for dyn ObjectTrait + '_ {} //~ ERROR lifetime bound not satisfied
    //~^ ERROR: cannot infer an appropriate lifetime for lifetime parameter `'_`

    fn use_it<'a>(val: &'a dyn ObjectTrait) -> impl OtherTrait<'a> + 'a {
        val.use_self() //~ ERROR borrowed data escapes
    }
}
mod u {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {}
    trait MyTrait {
        fn use_self(&self) -> &() where Self: 'static { panic!() }
    }
    trait Irrelevant {
        fn use_self(&self) -> &() { panic!() }
    }

    impl MyTrait for dyn ObjectTrait + '_ {}

    fn use_it<'a>(val: &'a dyn ObjectTrait) -> impl OtherTrait<'a> + 'a {
        val.use_self() //~ ERROR borrowed data escapes
    }
}
mod v {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {}
    trait MyTrait where Self: 'static {
        fn use_self(&'static self) -> &() { panic!() }
    }
    trait Irrelevant {
        fn use_self(&self) -> &() { panic!() }
    }

    impl MyTrait for dyn ObjectTrait {}

    fn use_it<'a>(val: &'a dyn ObjectTrait) -> impl OtherTrait<'a> + 'a {
        val.use_self() //~ ERROR borrowed data escapes
    }
}
mod w {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}

    trait Trait where Self: 'static { fn hello(&self) {} }

    impl Trait for dyn Foo + '_ { //~ERROR lifetime bound not satisfied
    //~^ ERROR: cannot infer an appropriate lifetime for lifetime parameter `'_`
        fn hello(&self) {}
    }
    fn convert<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR borrowed data escapes outside of function
    }
}
mod x {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + '_ where Self: '_ { //~ ERROR `'_` cannot be used here
        fn hello(&self) {}
    }
    fn convert<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello();
    }
}
fn main() {}
