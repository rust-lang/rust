//@ known-bug: #117392
pub trait BorrowComposite {
    type Ref<'a>: 'a;
}

impl BorrowComposite for () {
    type Ref<'a> = ();
}

pub trait Component<Args> {
    type Output;
}

impl<Args> Component<Args> for () {
    type Output = ();
}

pub fn delay<Args: BorrowComposite, Make: for<'a> FnMut(Args::Ref<'a>) -> C, C: Component<Args>>(
    make: Make,
) -> impl Component<Args> {
}

pub fn crash() -> impl Component<()> {
    delay(|()| delay(|()| ()))
}

pub fn main() {}
