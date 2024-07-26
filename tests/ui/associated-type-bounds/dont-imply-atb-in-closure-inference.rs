//@ check-pass

#![feature(type_alias_impl_trait)]

trait IsPtr {
    type Assoc;
}
impl<T> IsPtr for T {
    type Assoc = fn(i32);
}

type Tait = impl IsPtr<Assoc: Fn(i32)> + Fn(u32);

#[define_opaques(Tait)]
fn hello() {
    let _: Tait = |x| {};
}

fn main() {}
