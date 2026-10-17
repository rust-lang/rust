macro_rules! mn {
    (begin $($arg:ident),* end) => {
        [$($typo),*] //~ ERROR attempted to repeat an expression containing no syntax variables matched as repeating at this depth
        //~^ NOTE expected a repeatable metavariable: `$arg`
    };
}

macro_rules! mnr {
    (begin $arg:ident end) => { //~ NOTE this similarly named macro metavariable is unrepeatable
        [$($ard),*] //~ ERROR attempted to repeat an expression containing no syntax variables matched as repeating at this depth
        //~^ NOTE: argument not found
    };
}

macro_rules! err {
    (begin $arg:ident end) => {
        [$($typo),*] //~ ERROR attempted to repeat an expression containing no syntax variables matched as repeating at this depth
        //~^ NOTE this macro metavariable is not repeatable and there are no other repeatable metavariables
    };
}

fn main() {
    let x = 1;
    let _ = mn![begin x end];
    let _ = mnr![begin x end];
    let _ = err![begin x  end];
}
