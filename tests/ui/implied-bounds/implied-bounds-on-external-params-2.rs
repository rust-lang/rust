// A regression test for https://github.com/rust-lang/rust/issues/151637.

type Payload = Box<i32>;

trait StaticToStatic {
    fn static_to_static(self) -> &'static Payload
    where
        Self: 'static;
}
impl<'a> StaticToStatic for &'a Payload {
    fn static_to_static(self) -> &'static Payload
    where
        Self: 'static,
    {
        self // Legal. 'a must be 'static due to Self: 'static
    }
}

struct Wrap<T: StaticToStatic + 'static>(T);

trait ToStatic {
    fn to_static(self) -> &'static Payload;
}
impl<T: StaticToStatic> ToStatic for Wrap<T> {
    fn to_static(self) -> &'static Payload {
        self.0.static_to_static() // Legal. T: 'static is implied by Wrap<T>
    }
}

// Trait to allow mentioning FnOnce without mentioning the return type directly
trait MyFnOnce {
    type MyOutput;
    fn my_call_once(self) -> Self::MyOutput;
}
impl<F: FnOnce() -> T, T> MyFnOnce for F {
    type MyOutput = T;
    fn my_call_once(self) -> T {
        self()
    }
}

fn call<F: MyFnOnce<MyOutput: ToStatic>>(f: F) -> &'static Payload {
    f.my_call_once().to_static()
}

fn extend<T: StaticToStatic>(x: T) -> &'static Payload {
    let c = move || {
        //~^ ERROR: the parameter type `T` may not live long enough
        // Probably should be illegal, since Wrap requires T: 'static
        Wrap(x)
    };
    call(c)
}

fn main() {
    let x = Box::new(Box::new(1));
    let y = extend(&*x);
    drop(x);
    println!("{y}");
}
