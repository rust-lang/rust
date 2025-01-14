// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-enable-passes=+GVN,+SimplifyLocals-after-value-numbering
//@ edition: 2021

async fn inner() {
    panic!("disco");
}

// EMIT_MIR coroutine_dead_variants.outer-{closure#0}.SimplifyCfg-final.diff
async fn outer() {
    if false {
        inner().await;
    }
}
