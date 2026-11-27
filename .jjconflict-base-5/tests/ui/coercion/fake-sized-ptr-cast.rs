// Make sure borrowck doesn't ICE because it thinks a pointer cast is a metadata-preserving
// wide-to-wide ptr cast when it's actually (falsely) a wide-to-thin ptr cast due to an
// impossible dyn sized bound.

//@ check-pass

trait Trait<T> {}

fn func<'a>(x: *const (dyn Trait<()> + 'a))
where
    dyn Trait<u8> + 'a: Sized,
{
    let _x: *const dyn Trait<u8> = x as _;
}

fn main() {}
