#[allow(unused_macros)]
macro_rules! foo {
    ($id:ident) => {
        macro_rules! bar {
            ($id2:tt) => {
                #[cfg(any(target_feature = $id2, target_feature = $id2, target_feature = $id2, target_feature = $id2, target_feature = $id2))]
                fn $id() {}
            };
        }
    };
}

fn main() {}
