macro_rules! mac {
    ($attr_item: meta) => {
        #[cfg($attr_item)]
        struct S;
    }
}

mac!(an(arbitrary token stream));
//~^ ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `token`
//~| ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `stream`
//~| ERROR [E0537]

#[cfg(feature = -1)]
//~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `-`
fn handler() {}

fn main() {}
