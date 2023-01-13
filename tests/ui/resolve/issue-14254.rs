trait Foo {
    fn bar(&self);
    fn baz(&self) { }
    fn bah(_: Option<&Self>) { }
}

struct BarTy {
    x : isize,
    y : f64,
}

impl BarTy {
    fn a() {}
    fn b(&self) {}
}

impl Foo for *const BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        a;
        //~^ ERROR cannot find value `a`
    }
}

impl<'a> Foo for &'a BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        x;
        //~^ ERROR cannot find value `x`
        y;
        //~^ ERROR cannot find value `y`
        a;
        //~^ ERROR cannot find value `a`
        bah;
        //~^ ERROR cannot find value `bah`
        b;
        //~^ ERROR cannot find value `b`
    }
}

impl<'a> Foo for &'a mut BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        x;
        //~^ ERROR cannot find value `x`
        y;
        //~^ ERROR cannot find value `y`
        a;
        //~^ ERROR cannot find value `a`
        bah;
        //~^ ERROR cannot find value `bah`
        b;
        //~^ ERROR cannot find value `b`
    }
}

impl Foo for Box<BarTy> {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
    }
}

impl Foo for *const isize {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
    }
}

impl<'a> Foo for &'a isize {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
    }
}

impl<'a> Foo for &'a mut isize {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
    }
}

impl Foo for Box<isize> {
    fn bar(&self) {
        baz();
        //~^ ERROR cannot find function `baz`
        bah;
        //~^ ERROR cannot find value `bah`
    }
}

fn main() {}
