#![crate_name = "inner"]

pub struct SomeStruct;

fn asdf() {
    const _FOO: () = {
        impl Clone for SomeStruct {
            fn clone(&self) -> Self {
                SomeStruct
            }
        }
    };
}
