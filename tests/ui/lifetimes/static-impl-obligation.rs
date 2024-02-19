mod a {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo { //~ HELP consider relaxing the implicit `'static` requirement
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod b {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod c {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod d {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod e {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static { //~ HELP consider replacing this `'static` requirement
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod f {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod g {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod h {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod i {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod j {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod k {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod l {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo where Self: 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod m {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod n {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&'static self) {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod o {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&'static self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
mod p {
    trait Foo {}
    impl<'a> Foo for &'a u32 {}
    impl dyn Foo + 'static where Self: 'static {
        fn hello(&self) where Self: 'static {}
    }
    fn bar<'a>(x: &'a &'a u32) {
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
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
        let y: &dyn Foo = x; //~ ERROR lifetime may not live long enough
        y.hello();
    }
}
fn main() {}
