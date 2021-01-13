// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn foo<const LEN: usize, const DATA: [u8; LEN]>() {}
//~^ ERROR the type of const parameters must not
//[min]~^^ ERROR `[u8; _]` is forbidden as the type of a const generic parameter
fn main() {
    const DATA: [u8; 4] = *b"ABCD";
    foo::<4, DATA>();
    //[full]~^ ERROR constant expression depends on
}
