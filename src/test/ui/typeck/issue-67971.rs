struct S {}

fn foo(ctx: &mut S) -> String { //~ ERROR mismatched types
    // Don't suggest to remove semicolon as it won't fix anything
    ctx.sleep = 0;
    //~^ ERROR no field `sleep` on type `&mut S`
}

fn main() {}
