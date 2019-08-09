fn id<T>(x: T) -> T { x }

fn f() {
    let old = ['o'];         // statement 0
    let mut v1 = Vec::new(); // statement 1

    let mut v2 = Vec::new(); // statement 2

    {
        let young = ['y'];       // statement 3

        v2.push(&young[0]);      // statement 4
        //~^ ERROR `young[_]` does not live long enough
        //~| NOTE borrowed value does not live long enough
    } //~ NOTE `young[_]` dropped here while still borrowed

    let mut v3 = Vec::new(); // statement 5

    v3.push(&id('x'));           // statement 6
    //~^ ERROR temporary value dropped while borrowed
    //~| NOTE creates a temporary which is freed while still in use
    //~| NOTE temporary value is freed at the end of this statement
    //~| NOTE consider using a `let` binding to create a longer lived value

    {

        let mut v4 = Vec::new(); // (sub) statement 0

        v4.push(&id('y'));
        //~^ ERROR temporary value dropped while borrowed
        //~| NOTE creates a temporary which is freed while still in use
        //~| NOTE temporary value is freed at the end of this statement
        //~| NOTE consider using a `let` binding to create a longer lived value
        v4.use_ref();
        //~^ NOTE borrow later used here
    }                       // (statement 7)

    let mut v5 = Vec::new(); // statement 8

    v5.push(&id('z'));
    //~^ ERROR temporary value dropped while borrowed
    //~| NOTE creates a temporary which is freed while still in use
    //~| NOTE temporary value is freed at the end of this statement
    //~| NOTE consider using a `let` binding to create a longer lived value

    v1.push(&old[0]);

    (v1, v2, v3, /* v4 is above. */ v5).use_ref();
    //~^ NOTE borrow later used here
    //~| NOTE borrow later used here
    //~| NOTE borrow later used here
}

fn main() {
    f();
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
