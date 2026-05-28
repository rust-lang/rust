//@ known-bug: #117392
pub trait BorrowComposite {
    type Ref<'a>
    where
        Self: 'a;
}

impl BorrowComposite for () {
    type Ref<'a> = ();
}

pub trait Component<Args: BorrowComposite> {
    type Output;
}

impl<Args: BorrowComposite> Component<Args> for () {
    type Output = ();
}

struct Delay<Make> {
    _make: Make,
}

impl<
        Args: BorrowComposite,
        Make: for<'a> FnMut(Args::Ref<'a>) -> C,
        C: Component<Args>,
    > Component<Args> for Delay<Make>
{
    type Output = C::Output;
}

pub fn delay<
    Args: BorrowComposite,
    Make: for<'a> FnMut(Args::Ref<'a>) -> C,
    C: Component<Args>,
>(
    make: Make,
) -> impl Component<Args, Output = C::Output> {
    Delay { _make: make }
}

pub fn crash() -> impl Component<(), Output = ()> {
    delay(|()| delay(|()| ()))
}

pub fn main() {}
