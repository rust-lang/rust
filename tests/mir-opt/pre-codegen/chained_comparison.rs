// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=2

#![crate_type = "lib"]

pub struct Blueprint {
    pub fuel_tank_size: u32,
    pub payload: u32,
    pub wheel_diameter: u32,
    pub wheel_width: u32,
    pub storage: u32,
}

pub fn naive(a: &Blueprint, b: &Blueprint) -> bool {
    (a.fuel_tank_size == b.fuel_tank_size)
        && (a.payload == b.payload)
        && (a.wheel_diameter == b.wheel_diameter)
        && (a.wheel_width == b.wheel_width)
        && (a.storage == b.storage)
}

pub fn bitand(a: &Blueprint, b: &Blueprint) -> bool {
    (a.fuel_tank_size == b.fuel_tank_size)
        & (a.payload == b.payload)
        & (a.wheel_diameter == b.wheel_diameter)
        & (a.wheel_width == b.wheel_width)
        & (a.storage == b.storage)
}

pub fn returning(a: &Blueprint, b: &Blueprint) -> bool {
    if a.fuel_tank_size != b.fuel_tank_size {
        return false;
    }
    if a.payload != b.payload {
        return false;
    }
    if a.wheel_diameter != b.wheel_diameter {
        return false;
    }
    if a.wheel_width != b.wheel_width {
        return false;
    }
    if a.storage != b.storage {
        return false;
    }
    true
}

// EMIT_MIR chained_comparison.naive.PreCodegen.after.mir
// EMIT_MIR chained_comparison.bitand.PreCodegen.after.mir
// EMIT_MIR chained_comparison.returning.PreCodegen.after.mir
