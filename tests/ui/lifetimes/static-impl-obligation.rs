mod a {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo { //~ HELP consider relaxing the implicit `'static` requirement
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
    }
}
mod e {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static { //~ HELP consider replacing this `'static` requirement
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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
        y.hello(); //~ ERROR lifetime may not live long enough
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

    impl Trait for dyn Foo { //~ HELP consider relaxing the implicit `'static` requirement on the impl
        fn hello(&self) {}

    }
    fn convert<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x;
        y.hello(); //~ ERROR lifetime may not live long enough
    }
}
fn main() {}
