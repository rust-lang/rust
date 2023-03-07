#![feature(box_patterns)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]


#![deny(unreachable_patterns)]

mod foo {
    pub struct SecretlyEmpty {
        _priv: !,
    }
}

struct NotSoSecretlyEmpty {
    _priv: !,
}

fn foo() -> Option<NotSoSecretlyEmpty> {
    None
}

fn main() {
    let x: &[!] = &[];

    match x {
        &[]   => (),
        &[..] => (),    //~ ERROR unreachable pattern
    };

    let x: Result<Box<NotSoSecretlyEmpty>, &[Result<!, !>]> = Err(&[]);
    match x {
        Ok(box _) => (),    //~ ERROR unreachable pattern
        Err(&[]) => (),
        Err(&[..]) => (),   //~ ERROR unreachable pattern
    }

    let x: Result<foo::SecretlyEmpty, Result<NotSoSecretlyEmpty, u32>> = Err(Err(123));
    match x {
        Ok(_y) => (),
        Err(Err(_y)) => (),
        Err(Ok(_y)) => (),  //~ ERROR unreachable pattern
    }

    while let Some(_y) = foo() {
        //~^ ERROR unreachable pattern
    }
}
