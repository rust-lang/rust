mod hidden {
    pub struct Record { secret: u8 }
}

fn leak(hidden::Record { secret }: hidden::Record) -> u8 {
    //~^ ERROR field `secret` of struct `Record` is private
    secret
}

fn f() -> [u8; { let _ = hidden::Record { secret: 1 }; 0 }] {
    //~^ ERROR field `secret` of struct `Record` is private
    []
}

trait Test {
    fn trait_method() -> [u8; { let _ = hidden::Record { secret: 1 }; 0 }];
    //~^ ERROR field `secret` of struct `Record` is private
}

fn main() {}
