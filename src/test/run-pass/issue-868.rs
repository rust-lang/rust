fn f<T>(g: fn() -> T) -> T { g() }

fn main() {
  let _x = f( | | { 10 });
    // used to be: cannot determine a type for this expression
    f(| | { });
    // ditto
    f( | | { ()});
    // always worked
    let _: () = f(| | { });
    // empty block with no type info should compile too
    let _ = f(||{});
    let _ = (||{});
}

