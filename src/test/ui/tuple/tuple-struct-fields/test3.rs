macro_rules! define_struct {
    ($t:ty) => {
        struct S1(pub($t));
        struct S2(pub (in foo) ());
        struct S3(pub($t) ());
        //~^ ERROR expected one of `)` or `,`, found `(`
    }
}

mod foo {
    define_struct! { foo } //~ ERROR cannot find type `foo` in this scope
}

fn main() {}
