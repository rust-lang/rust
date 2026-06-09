//@ run-rustfix
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn a() {
    become (|| ())();
    //~^ ERROR: tail calling closures directly is not allowed
}

fn aa((): ()) {
    become (|()| ())(());
    //~^ ERROR: tail calling closures directly is not allowed
}

fn aaa((): (), _: i32) {
    become (|(), _| ())((), 1);
    //~^ ERROR: tail calling closures directly is not allowed
}

fn v((): (), ((), ()): ((), ())) -> (((), ()), ()) {
    let f = |(), ((), ())| (((), ()), ());
    become f((), ((), ()));
    //~^ ERROR: tail calling closures directly is not allowed
}

fn main() {
    a();
    aa(());
    aaa((), 1);
    v((), ((), ()));
}
