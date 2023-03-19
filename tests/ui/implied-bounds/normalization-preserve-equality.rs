// check-pass
// issue: #106569

struct Equal<'a, 'b>(&'a &'b (), &'b &'a ()); // implies 'a == 'b

trait Trait { type Ty; }

impl<'x> Trait for Equal<'x, 'x> { type Ty = (); }

fn test1<'a, 'b>(_: (<Equal<'a, 'b> as Trait>::Ty, Equal<'a, 'b>)) {
    let _ = None::<Equal<'a, 'b>>;
}

fn test2<'a, 'b>(_: <Equal<'a, 'b> as Trait>::Ty, _: Equal<'a, 'b>) {
    let _ = None::<Equal<'a, 'b>>;
}

fn main() {}
