// Test that `binding @ subpat` acts as a product context with respect to duplicate binding names.
// The code that is tested here lives in resolve (see `resolve_pattern_inner`).


fn main() {
    fn f(a @ a @ a: ()) {}
    //~^ ERROR identifier `a` is bound more than once in this parameter list
    //~| ERROR identifier `a` is bound more than once in this parameter list

    match Ok(0) {
        Ok(a @ b @ a)
        //~^ ERROR identifier `a` is bound more than once in the same pattern
        | Err(a @ b @ a) //~ ERROR cannot assign twice to immutable variable `a`
        //~^ ERROR identifier `a` is bound more than once in the same pattern
        => {}
    }

    let a @ a @ a = ();
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern
    let ref a @ ref a = ();
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    let ref mut a @ ref mut a = (); //~ ERROR cannot borrow value as mutable more than once at a time
    //~^ ERROR identifier `a` is bound more than once in the same pattern

    let a @ (Ok(a) | Err(a)) = Ok(());
    //~^ ERROR identifier `a` is bound more than once in the same pattern
    //~| ERROR identifier `a` is bound more than once in the same pattern
}
