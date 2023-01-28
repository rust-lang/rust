//check-fail

#![crate_type="lib"]

fn r({) {
    Ok {             //~ ERROR mismatched types [E0308]
        d..||_=m
    }
}
//~^^^^^ ERROR expected parameter name, found `{`
//~| ERROR expected one of `,`, `:`, or `}`, found `..`
//~^^^^^ ERROR cannot find value `d` in this scope [E0425]
//~| ERROR cannot find value `m` in this scope [E0425]
//~| ERROR variant `Result<_, _>::Ok` has no field named `d` [E0559]
