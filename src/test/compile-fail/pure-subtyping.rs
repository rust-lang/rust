// Test rules governing higher-order pure fns.

fn assign_to_pure(x: pure fn(), y: fn(), z: unsafe fn()) {
    let a: pure fn() = x;
    let b: pure fn() = y; //~ ERROR expected pure fn but found impure fn
    let c: pure fn() = z; //~ ERROR expected pure fn but found unsafe fn
}

fn assign_to_impure(x: pure fn(), y: fn(), z: unsafe fn()) {
    let h: fn() = x;
    let i: fn() = y;
    let j: fn() = z; //~ ERROR expected impure fn but found unsafe fn
}

fn assign_to_unsafe(x: pure fn(), y: fn(), z: unsafe fn()) {
    let m: unsafe fn() = x;
    let n: unsafe fn() = y;
    let o: unsafe fn() = z;
}

fn assign_to_pure2(x: pure fn@(), y: fn@(), z: unsafe fn@()) {
    let a: pure fn() = x;
    let b: pure fn() = y; //~ ERROR expected pure fn but found impure fn
    let c: pure fn() = z; //~ ERROR expected pure fn but found unsafe fn

    let a: pure fn~() = x; //~ ERROR fn storage differs: expected ~ but found @
    let b: pure fn~() = y; //~ ERROR fn storage differs: expected ~ but found @
    let c: pure fn~() = z; //~ ERROR fn storage differs: expected ~ but found @

    let a: unsafe fn~() = x; //~ ERROR fn storage differs: expected ~ but found @
    let b: unsafe fn~() = y; //~ ERROR fn storage differs: expected ~ but found @
    let c: unsafe fn~() = z; //~ ERROR fn storage differs: expected ~ but found @
}

fn main() {
}
