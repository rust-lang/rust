// run-pass
// Test that we can use method notation to call methods based on a
// projection bound from a trait. Issue #20469.

///////////////////////////////////////////////////////////////////////////


trait MakeString {
    fn make_string(&self) -> String;
}

impl MakeString for isize {
    fn make_string(&self) -> String {
        format!("{}", *self)
    }
}

impl MakeString for usize {
    fn make_string(&self) -> String {
        format!("{}", *self)
    }
}

///////////////////////////////////////////////////////////////////////////

trait Foo {
    type F: MakeString;

    fn get(&self) -> &Self::F;
}

fn foo<F:Foo>(f: &F) -> String {
    f.get().make_string()
}

///////////////////////////////////////////////////////////////////////////

struct SomeStruct {
    field: isize,
}

impl Foo for SomeStruct {
    type F = isize;

    fn get(&self) -> &isize {
        &self.field
    }
}

///////////////////////////////////////////////////////////////////////////

struct SomeOtherStruct {
    field: usize,
}

impl Foo for SomeOtherStruct {
    type F = usize;

    fn get(&self) -> &usize {
        &self.field
    }
}

fn main() {
    let x = SomeStruct { field: 22 };
    assert_eq!(foo(&x), format!("22"));

    let x = SomeOtherStruct { field: 44 };
    assert_eq!(foo(&x), format!("44"));
}
