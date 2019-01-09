pub static X: usize = 1;

fn main() {
    match 1 {
        self::X => { },
        //~^ ERROR expected unit struct/variant or constant, found static `self::X`
        _       => { },
    }
}
