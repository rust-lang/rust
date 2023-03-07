// run-pass
trait Trait<const N: u8 = 12> {
    fn uwu(&self) -> u8 {
        N
    }
}

impl Trait for u32 {}

impl Trait<12> for u64 {
    fn uwu(&self) -> u8 {
        *self as u8
    }
}

fn foo(arg: &dyn Trait) -> u8 {
    arg.uwu()
}

trait Traitor<const N: u8 = 1, const M: u8 = N> {
    fn owo(&self) -> u8 {
        M
    }
}

impl Traitor<2> for bool { }
impl Traitor for u8 {
    fn owo(&self) -> u8 {
        *self
    }
}

fn bar<const N: u8>(arg: &dyn Traitor<N>) -> u8 {
    arg.owo()
}

fn main() {
    assert_eq!(foo(&10_u32), 12);
    assert_eq!(foo(&3_u64), 3);

    assert_eq!(bar(&true), 2);
    assert_eq!(bar(&1_u8), 1);
}
