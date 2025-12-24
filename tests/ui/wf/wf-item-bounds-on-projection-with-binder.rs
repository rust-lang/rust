//@ check-pass
//@ compile-flags:  -Zstrict-projection-item-bounds

// Since we don't have nested binders, we just concatenate the binder of item bounds to the
// projection predicate's.

trait Higher1<'a> {}

trait HasAssoc1<'b> {
    type Assoc: for<'a> Higher1<'a>;
}

fn imply1<T, U>()
where
    // Elaborated: `U: for<'b, 'a> Higher1<'a>
    T: for<'b> HasAssoc1<'b, Assoc = U>,
{
    assert_higher1::<U>();
}
fn assert_higher1<T: for<'b, 'a> Higher1<'a>>() {}

// Actually use two bound regions.
trait Higher2<'a, 'b> {}

trait HasAssoc2<'b> {
    type Assoc: for<'a> Higher2<'a, 'b>;
}

fn imply2<T, U>()
where
    // Elaborated: `U: for<'b, 'a> Higher2<'a, 'b>`
    T: for<'b> HasAssoc2<'b, Assoc = U>,
{
    assert_higher2::<U>();
}
fn assert_higher2<U: for<'b, 'a> Higher2<'a, 'b>>() {}

// Item bound contains nested binders.
trait Higher3<'a, 'b, T> {}

trait HasAssoc3 {
    type Assoc<'a>: for<'b> Higher3<'a, 'b, for<'c> fn(&'c str, &'b str)>;
}

fn imply3<T, U>()
where
    // Elaborated: `U: for<'a, 'b> Higher3<'a, 'b, for<'c> fn(&'c str, &'b str)>`
    T: for<'a> HasAssoc3<Assoc<'a> = U>,
{
    assert_higher3::<U>();
}

fn assert_higher3<U: for<'a, 'b> Higher3<'a, 'b, for<'c> fn(&'c str, &'b str)>>() {}

fn main() {}
