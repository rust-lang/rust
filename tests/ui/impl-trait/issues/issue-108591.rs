//@ check-pass

#![feature(type_alias_impl_trait)]

struct MyTy<'a>(Vec<u8>, &'a ());

impl MyTy<'_> {
    fn one(&mut self) -> &mut impl Sized {
        &mut self.0
    }
    fn two(&mut self) -> &mut (impl Sized + 'static) {
        self.one()
    }
}

type Opaque2 = impl Sized;
type Opaque<'a> = Opaque2;
#[define_opaque(Opaque)]
fn define<'a>() -> Opaque<'a> {}

fn test<'a>() {
    None::<&'static Opaque<'a>>;
}

fn one<'a, 'b: 'b>() -> &'a impl Sized {
    &()
}
fn two<'a, 'b>() {
    one::<'a, 'b>();
}

fn main() {}
