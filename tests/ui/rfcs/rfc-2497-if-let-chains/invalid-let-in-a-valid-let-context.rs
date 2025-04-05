#![feature(let_chains)]

fn main() {
    let _opt = Some(1i32);

    #[cfg(false)]
    {
        let _ = &&let Some(x) = Some(42);
        //~^ ERROR expected expression, found `let` statement
    }
    #[cfg(false)]
    {
        if let Some(elem) = _opt && [1, 2, 3][let _ = &&let Some(x) = Some(42)] = 1 {
        //~^ ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement
            true
        }
    }

    #[cfg(false)]
    {
        if let Some(elem) = _opt && {
            [1, 2, 3][let _ = ()];
            //~^ ERROR expected expression, found `let` statement
            true
        } {
        }
    }

    #[cfg(false)]
    {
        if let Some(elem) = _opt && [1, 2, 3][let _ = ()] = 1 {
        //~^ ERROR expected expression, found `let` statement
        //~| ERROR expected expression, found `let` statement
            true
        }
    }
    #[cfg(false)]
    {
        if let a = 1 && {
            let x = let y = 1;
            //~^ ERROR expected expression, found `let` statement
        } {
        }
    }
}
