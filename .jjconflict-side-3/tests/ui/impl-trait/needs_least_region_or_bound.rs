//@ check-pass

trait MultiRegionTrait<'a, 'b> {}
impl<'a, 'b> MultiRegionTrait<'a, 'b> for (&'a u32, &'b u32) {}

fn no_least_region<'a, 'b>(x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> {
    // Here we have a constraint that:
    //
    // (x, y) has type (&'0 u32, &'1 u32)
    //
    // where
    //
    // 'a: '0
    //
    // then we require that `('0 u32, &'1 u32): MultiRegionTrait<'a,
    // 'b>`, which winds up imposing a requirement that `'0 = 'a` and
    // `'1 = 'b`.
    (x, y)
}

fn main() {}
