#![feature(catch_expr)]

fn main() {
    let x = do catch {
        foo()?
    };

    let x = do catch /* Invisible comment */ { foo()? };

    let x = do catch {
        unsafe { foo()? }
    };

    let y = match (do catch {
        foo()?
    }) {
        _ => (),
    };

    do catch {
        foo()?;
    };

    do catch {
        // Regular do catch block
    };
}
