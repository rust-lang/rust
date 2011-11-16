fn magic(+x: {a: @int}) { log x; }
fn magic2(+x: @int) { log x; }

fn main() {
    let a = {a: @10}, b = @10;
    magic(a); magic({a: @20});
    magic2(b); magic2(@20);
}
