// compile-flags: -Ztrait-solver=next
trait Relate {}

struct Outlives<'a, 'b>(&'a u8, &'b u8);
impl<'a, 'b> Relate for Outlives<'a, 'b> where 'a: 'b, {}

struct Equal<'a, 'b>(&'a u8, &'b u8);
impl<'a> Relate for Equal<'a, 'a> {}

macro_rules! rule {
    ( $name:ident<$($lt:lifetime),*> :- $( ($($bound:tt)*) ),* ) => {
        struct $name<$($lt),*>($(&$lt u8),*);
        impl<$($lt),*> $crate::Relate for $name<$($lt),*>
            where $( $($bound)*: $crate::Relate, )*
        {}
    };
}

// ----

trait CoherenceTrait {}
impl<T> CoherenceTrait for T {}

macro_rules! assert_false_by_leak_check {
    ( exist<$($lt:lifetime),*> $( ($($bound:tt)*) ),* ) => {
        struct Unique<$($lt),*>($(&$lt u8),*);
        impl<$($lt),*> $crate::CoherenceTrait for Unique<$($lt),*>
            //~^ ERROR for type `test1::Unique`
            //~| ERROR for type `test3::Unique`
            //~| ERROR for type `test6::Unique`
            where $( $($bound)*: $crate::Relate, )*
        {}
    };
}

mod test1 {
    use super::*;
    assert_false_by_leak_check!(
        exist<> (for<'a, 'b> Outlives<'a, 'b>)
    );
}

mod test2 {
    use super::*;
    assert_false_by_leak_check!(
        exist<'a> (for<'b> Outlives<'b, 'a>)
    );
}

mod test3 {
    use super::*;
    rule!( OutlivesPlaceholder<'a> :- (for<'b> Outlives<'a, 'b>) );
    assert_false_by_leak_check!(
        exist<> (for<'a> OutlivesPlaceholder<'a>)
    );
}

mod test4 {
    use super::*;
    rule!( OutlivedByPlaceholder<'a> :- (for<'b> Outlives<'b, 'a>) );
    assert_false_by_leak_check!(
        exist<> (for<'a> OutlivedByPlaceholder<'a>)
    );
}

mod test5 {
    use super::*;
    rule!( EqualsPlaceholder<'a> :- (for<'b> Equal<'b, 'a>) );
    assert_false_by_leak_check!(
        exist<> (for<'a> EqualsPlaceholder<'a>)
    );
}

mod test6 {
    use super::*;
    assert_false_by_leak_check!(
        exist<> (for<'a, 'b> Equal<'a, 'b>)
    );
}

fn main() {}
