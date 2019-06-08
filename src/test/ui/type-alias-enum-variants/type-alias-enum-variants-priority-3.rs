enum E {
    V
}

fn check() -> <E>::V {}
//~^ ERROR expected type, found variant `V`

fn main() {}
