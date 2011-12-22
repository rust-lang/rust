fn magic(+x: {a: @int}) { log_full(core::debug, x); }
fn magic2(+x: @int) { log_full(core::debug, x); }

fn main() {
    let a = {a: @10}, b = @10;
    magic(a); magic({a: @20});
    magic2(b); magic2(@20);
}
