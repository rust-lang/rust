//@ check-pass

#![warn(dropping_references)]

struct SomeStruct;

fn main() {
    drop(&SomeStruct); //~ WARN calls to `std::mem::drop`

    let mut owned1 = SomeStruct;
    drop(&owned1); //~ WARN calls to `std::mem::drop`
    drop(&&owned1); //~ WARN calls to `std::mem::drop`
    drop(&mut owned1); //~ WARN calls to `std::mem::drop`
    drop(owned1);

    let reference1 = &SomeStruct;
    drop(reference1); //~ WARN calls to `std::mem::drop`

    let reference2 = &mut SomeStruct;
    drop(reference2); //~ WARN calls to `std::mem::drop`

    let ref reference3 = SomeStruct;
    drop(reference3); //~ WARN calls to `std::mem::drop`
}

#[allow(dead_code)]
fn test_generic_fn_drop<T>(val: T) {
    drop(&val); //~ WARN calls to `std::mem::drop`
    drop(val);
}

#[allow(dead_code)]
fn test_similarly_named_function() {
    fn drop<T>(_val: T) {}
    drop(&SomeStruct); //OK; call to unrelated function which happens to have the same name
    std::mem::drop(&SomeStruct); //~ WARN calls to `std::mem::drop`
}

#[derive(Copy, Clone)]
pub struct Error;
fn produce_half_owl_error() -> Result<(), Error> {
    Ok(())
}

fn produce_half_owl_ok() -> Result<bool, ()> {
    Ok(true)
}

#[allow(dead_code)]
fn test_owl_result() -> Result<(), ()> {
    produce_half_owl_error().map_err(|_| ())?;
    produce_half_owl_ok().map(|_| ())?;
    // the following should not be linted,
    // we should not force users to use toilet closures
    // to produce owl results when drop is more convenient
    produce_half_owl_error().map_err(drop)?;
    produce_half_owl_ok().map_err(drop)?;
    Ok(())
}

#[allow(dead_code)]
fn test_owl_result_2() -> Result<u8, ()> {
    produce_half_owl_error().map_err(|_| ())?;
    produce_half_owl_ok().map(|_| ())?;
    // the following should not be linted,
    // we should not force users to use toilet closures
    // to produce owl results when drop is more convenient
    produce_half_owl_error().map_err(drop)?;
    produce_half_owl_ok().map(drop)?;
    Ok(1)
}

#[allow(unused)]
#[allow(clippy::unit_cmp)]
fn issue10122(x: u8) {
    // This is a function which returns a reference and has a side-effect, which means
    // that calling drop() on the function is considered an idiomatic way of achieving
    // the side-effect in a match arm.
    fn println_and<T>(t: &T) -> &T {
        println!("foo");
        t
    }

    match x {
        // Don't lint (copy type), we only care about side-effects
        0 => drop(println_and(&12)),
        // Don't lint (no copy type), we only care about side-effects
        1 => drop(println_and(&String::new())),
        2 => {
            // Lint, even if we only care about the side-effect, it's already in a block
            drop(println_and(&13)); //~ WARN calls to `std::mem::drop`
        },
        // Lint, idiomatic use is only in body of `Arm`
        3 if drop(println_and(&14)) == () => (), //~ WARN calls to `std::mem::drop`
         // Lint, not a fn/method call
        4 => drop(&2), //~ WARN calls to `std::mem::drop`
        _ => (),
    }
}

fn issue112653() {
    fn foo() -> Result<&'static u8, ()> {
        println!("doing foo");
        Ok(&0) // result is not always useful, the side-effect matters
    }
    fn bar() {
        println!("doing bar");
    }

    fn stuff() -> Result<(), ()> {
        match 42 {
            0 => drop(foo()?),  // drop is needed because we only care about side-effects
            1 => bar(),
            _ => (),  // doing nothing (no side-effects needed here)
        }
        Ok(())
    }
}
