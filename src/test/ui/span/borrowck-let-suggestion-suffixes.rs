#![feature(rustc_attrs)]
fn id<T>(x: T) -> T { x }

fn f() {
    let old = ['o'];         // statement 0
    let mut v1 = Vec::new(); // statement 1

    let mut v2 = Vec::new(); // statement 2

    let young = ['y'];       // statement 3

    v2.push(&young[0]);      // statement 4
    //~^ ERROR `young[..]` does not live long enough
    //~| NOTE borrowed value does not live long enough
    //~| NOTE values in a scope are dropped in the opposite order they are created

    let mut v3 = Vec::new(); // statement 5

    v3.push(&id('x'));           // statement 6
    //~^ ERROR borrowed value does not live long enough
    //~| NOTE temporary value does not live long enough
    //~| NOTE temporary value dropped here while still borrowed
    //~| NOTE consider using a `let` binding to increase its lifetime

    {

        let mut v4 = Vec::new(); // (sub) statement 0

        v4.push(&id('y'));
        //~^ ERROR borrowed value does not live long enough
        //~| NOTE temporary value does not live long enough
        //~| NOTE temporary value dropped here while still borrowed
        //~| NOTE consider using a `let` binding to increase its lifetime
        v4.use_ref();
    }                       // (statement 7)
    //~^ NOTE temporary value needs to live until here

    let mut v5 = Vec::new(); // statement 8

    v5.push(&id('z'));
    //~^ ERROR borrowed value does not live long enough
    //~| NOTE temporary value does not live long enough
    //~| NOTE temporary value dropped here while still borrowed
    //~| NOTE consider using a `let` binding to increase its lifetime

    v1.push(&old[0]);

    (v1, v2, v3, /* v4 is above. */ v5).use_ref();
}
//~^ NOTE `young[..]` dropped here while still borrowed
//~| NOTE temporary value needs to live until here
//~| NOTE temporary value needs to live until here

fn main() { #![rustc_error] // rust-lang/rust#49855
    f();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
