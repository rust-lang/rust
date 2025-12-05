#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(tuple_trait)]

fn foo<F: Fn<T>, T:std::marker::Tuple>(f: Option<F>, t: T) {
    let y = (f.unwrap()).call(t);
}

fn main() {
    foo::<fn() -> str, _>(None, ());
    //~^ ERROR the size for values of type `str` cannot be known at compilation time

    foo::<for<'a> fn(&'a ()) -> (dyn std::fmt::Display + 'a), _>(None, (&(),));
    //~^ ERROR the size for values of type `(dyn std::fmt::Display + 'a)` cannot be known at compilation time
}
