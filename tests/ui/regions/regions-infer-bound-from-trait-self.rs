// Test that we can derive lifetime bounds on `Self` from trait
// inheritance.

trait Static : 'static { }

trait Is<'a> : 'a { }

struct Inv<'a> {
    x: Option<&'a mut &'a isize>
}

fn check_bound<'a,A:'a>(x: Inv<'a>, a: A) { }

// In these case, `Self` inherits `'static`.

trait InheritsFromStatic : Sized + 'static {
    fn foo1<'a>(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}
trait InheritsFromStaticIndirectly : Sized + Static {
    fn foo1<'a>(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}


// In these case, `Self` inherits `'a`.

trait InheritsFromIs<'a> : Sized + 'a {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}

trait InheritsFromIsIndirectly<'a> : Sized + Is<'a> {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}

// In this case, `Self` inherits nothing.

trait InheritsFromNothing<'a> : Sized {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
            //~^ ERROR parameter type `Self` may not live long enough
    }
}

fn main() { }
