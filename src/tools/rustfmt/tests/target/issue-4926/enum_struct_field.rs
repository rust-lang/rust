// rustfmt-struct_field_align_threshold: 30
// rustfmt-enum_discrim_align_threshold: 30
// rustfmt-imports_layout: HorizontalVertical

#[derive(Default)]
struct InnerStructA {
    bbbbbbbbb: i32,
    cccccccc:  i32,
}

enum SomeEnumNamedD {
    E(InnerStructA),
    F {
        ggggggggggggggggggggggggg: bool,
        h:                         bool,
    },
}

impl SomeEnumNamedD {
    fn f_variant() -> Self {
        Self::F {
            ggggggggggggggggggggggggg: true,
            h:                         true,
        }
    }
}

fn main() {
    let kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk = SomeEnumNamedD::f_variant();
    let something_we_care_about = matches!(
        kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk,
        SomeEnumNamedD::F {
            ggggggggggggggggggggggggg: true,
            ..
        }
    );

    if something_we_care_about {
        println!("Yup it happened");
    }
}
