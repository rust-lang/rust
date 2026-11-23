use std::mem::type_info::TypeKind;
//~^ ERROR: use of unstable library feature `type_info`

fn main() {
    let ty = std::mem::type_info::TypeKind::of::<()>();
    //~^ ERROR: use of unstable library feature `type_info`
    //~| ERROR: use of unstable library feature `type_info`
}
