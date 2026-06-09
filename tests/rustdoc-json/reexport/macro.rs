//@ edition:2018

//@ set repro_id = "$.index[?(@.name=='repro')].id"
#[macro_export]
macro_rules! repro {
    () => {};
}

//@ set repro2_id = "$.index[?(@.docs=='Re-export')].id"
/// Re-export
pub use crate::repro as repro2;

//@ ismany "$.index[?(@.name=='macro')].inner.module.items[*]" $repro_id $repro2_id
