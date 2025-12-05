//@ build-fail
//~^ ERROR cycle detected when computing layout of `Wrapper<()>`

trait Trait {
    type Assoc;
}

impl Trait for () {
    type Assoc = Wrapper<()>;
}

struct Wrapper<T: Trait> {
    _x: <T as Trait>::Assoc,
}

fn abi<T: Trait>(_: Option<Wrapper<T>>) {}

fn indirect<T: Trait>() {
    abi::<T>(None);
}

fn main() {
    indirect::<()>();
}
