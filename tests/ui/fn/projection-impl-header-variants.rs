// Regression test variants for #100051 — projection impl header implied bounds

trait MyTrait { type Assoc; }
impl<T> MyTrait for T { type Assoc = (); }

trait SoundTrait { type Assoc; }
impl SoundTrait for i32 { type Assoc = i32; }

trait Convert<'a, 'b> {
    fn convert(self, s: &'a str) -> &'b str;
}

// Unsound: projection normalizes away lifetime constraint
impl<'a, 'b> Convert<'a, 'b> for <&'b &'a () as MyTrait>::Assoc
where
    for<'x, 'y> &'x &'y (): MyTrait,
{
    fn convert(self, s: &'a str) -> &'b str {
        s //~ ERROR lifetime may not live long enough
    }
}

// Sound: no lifetime-carrying types in projection
impl<'a> Convert<'a, 'a> for <i32 as SoundTrait>::Assoc {
    fn convert(self, s: &'a str) -> &'a str {
        s  // OK — same lifetime
    }
}

fn main() {}
