// Regression test. This didn't valgrind after we stopped zeroing the exchange heap

pub struct VariantDoc {
    desc: Option<~str>,
    sig: Option<~str>
}

fn main() {
    let variants = ~[
        VariantDoc {
            desc: None,
            sig: None
        }
    ];

    let _vdoc = do vec::map(variants) |variant| {
        VariantDoc {
            desc: None,
            .. copy *variant
        }
    };
}
