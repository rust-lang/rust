//@ check-pass
#![feature(macro_metavar_expr_concat)]

struct A;
struct B;
const AA: A = A;
const BB: B = B;

macro_rules! define_ioctl_data {
    (struct $s:ident {
        $($field:ident: $ty:ident $([$opt:ident])?,)*
    }) => {
        pub struct $s {
            $($field: $ty,)*
        }

        impl $s {
            $($(
                fn ${concat(get_, $field)}(&self) -> $ty {
                    let _ = $opt;
                    todo!()
                }
            )?)*
        }
    };
}

define_ioctl_data! {
    struct Foo {
        a: A [AA],
        b: B [BB],
    }
}

fn main() {}
