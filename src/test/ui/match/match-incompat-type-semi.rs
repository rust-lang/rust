// Diagnostic enhancement explained in issue #75418.
// Point at the last statement in the block if there's no tail expression,
// and suggest removing the semicolon if appropriate.

fn main() {
    let _ = match Some(42) {
        Some(x) => {
            x
        },
        None => {
            0;
            //~^ ERROR incompatible types
            //~| HELP consider removing this semicolon
        },
    };

    let _ = if let Some(x) = Some(42) {
        x
    } else {
        0;
        //~^ ERROR incompatible types
        //~| HELP consider removing this semicolon
    };

    let _ = match Some(42) {
        Some(x) => {
            x
        },
        None => {
            ();
            //~^ ERROR incompatible types
        },
    };

    let _ = match Some(42) {
        Some(x) => {
            x
        },
        None => { //~ ERROR incompatible types
        },
    };
}
