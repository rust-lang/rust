fn of<T>() -> @fn(T) { fail; }
fn subtype<T>(x: @fn(T)) { fail; }

fn test_fn<T>(_x: &x/T, _y: &y/T, _z: &z/T) {
    // Here, x, y, and z are free.  Other letters
    // are bound.  Note that the arrangement
    // subtype::<T1>(of::<T2>()) will typecheck
    // iff T1 <: T2.

    // should be the default:
    subtype::<@static/fn()>(of::<@fn()>());
    subtype::<@fn()>(of::<@static/fn()>());

    //
    subtype::<@x/fn()>(of::<@fn()>());    //~ ERROR mismatched types
    subtype::<@x/fn()>(of::<@y/fn()>());  //~ ERROR mismatched types

    subtype::<@x/fn()>(of::<@static/fn()>()); //~ ERROR mismatched types
    subtype::<@static/fn()>(of::<@x/fn()>());

}
