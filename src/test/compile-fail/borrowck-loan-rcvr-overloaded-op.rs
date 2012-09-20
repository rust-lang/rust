// xfail-test
// xfail-fast

// XFAIL'd because of error message problems with demoded Add.

struct Point { 
    x: int,
    y: int,
}

#[cfg(stage0)]
impl Point : ops::Add<int,int> {
    pure fn add(&&z: int) -> int {
        self.x + self.y + z
    }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl Point : ops::Add<int,int> {
    pure fn add(z: &int) -> int {
        self.x + self.y + (*z)
    }
}

impl Point {
    fn times(z: int) -> int {
        self.x * self.y * z
    }
}

fn a() {
    let mut p = Point {x: 3, y: 4};

    // ok (we can loan out rcvr)
    p + 3;
    p.times(3);
}

fn b() {
    let mut p = Point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let q = &mut p; //~ NOTE prior loan as mutable granted here
    //~^ NOTE prior loan as mutable granted here

    p + 3; //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
    p.times(3); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan

    q.x += 1;
}

fn c() {
    // Here the receiver is in aliased memory and hence we cannot
    // consider it immutable:
    let q = @mut Point {x: 3, y: 4};

    // ...this is ok for pure fns
    *q + 3;


    // ...but not impure fns
    (*q).times(3); //~ ERROR illegal borrow unless pure
    //~^ NOTE impure due to access to impure function
}

fn main() {
}

