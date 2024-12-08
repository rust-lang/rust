//@ check-pass

trait A {
    type T: B<U<1i32> = ()>;
}

trait B {
    type U<const C: i32>;
}

fn f<T: A>() {
    let _: <<T as A>::T as B>::U<1i32> = ();
}

fn main() {}
