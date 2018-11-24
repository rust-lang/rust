const fn mutable_ref_in_const() -> u8 {
    let mut a = 0;
    let b = &mut a; //~ ERROR mutable references in const fn are unstable
    *b
}

struct X;

impl X {
    const fn inherent_mutable_ref_in_const() -> u8 {
        let mut a = 0;
        let b = &mut a; //~ ERROR mutable references in const fn are unstable
        *b
    }
}

static mut FOO: u32 = {
    let mut a = 0;
    let b = &mut a; //~ references in statics may only refer to immutable
    *b
};

static mut BAR: Option<String> = {
    let mut a = None;
    // taking a mutable reference erases everything we know about `a`
    { let b = &mut a; }  //~ references in statics may only refer to immutable
    a
};

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

static mut BAR2: &mut Option<Foo> = {
    let mut a = &mut None; //~ references in statics may only refer to immutable values
    //~^ does not live long enough
    *a = Some(Foo); //~ unimplemented expression type
    a
};

fn main() {}
