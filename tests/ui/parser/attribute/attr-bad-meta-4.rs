macro_rules! mac {
    ($attr_item: meta) => {
        #[cfg($attr_item)]
        //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `meta` metavariable
        struct S;
    }
}

mac!(an(arbitrary token stream));

#[cfg(feature = -1)]
//~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `-`
fn handler() {}

fn main() {}
