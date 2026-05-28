struct Foo<T>(T);

trait GoodBye {
    type Forget;
}
impl<T> GoodBye for T {
    type Forget = ();
}

trait NeedsWf<'a, 'b> {
    type Assoc;
}

impl<'a, 'b> NeedsWf<'a, 'b> for Foo<<&'a &'b () as GoodBye>::Forget> {
    type Assoc = &'a &'b ();
    //~^ ERROR in type `&'a &'b ()`, reference has a longer lifetime than the data it references
}

fn needs_wf<'a, 'b, T: NeedsWf<'a, 'b>>() {}

fn foo<'a: 'a, 'b: 'b>(_: &'b String) {
    needs_wf::<'a, 'b, Foo<()>>();
}

fn main() {
    let x = String::from("hello");
    foo::<'static, '_>(&x);
}
