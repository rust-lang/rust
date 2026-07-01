//@ needs-unwind
//@ skip-filecheck

#[inline(never)]
pub fn make_one(x: u64) -> Result<String, ()> {
    if x == u64::MAX { Err(()) } else { Ok(x.to_string()) }
}

pub struct Big {
    f0: String,
    f1: String,
    f2: String,
    f3: String,
    f4: String,
}

// EMIT_MIR fallible_struct_drop.build.built.after.mir
// EMIT_MIR fallible_struct_drop.build.ElaborateDrops.diff
#[inline(never)]
pub fn build(s: u64) -> Result<Big, ()> {
    Ok(Big {
        f0: make_one(s ^ 0)?,
        f1: make_one(s ^ 1)?,
        f2: make_one(s ^ 2)?,
        f3: make_one(s ^ 3)?,
        f4: make_one(s ^ 4)?,
    })
}
