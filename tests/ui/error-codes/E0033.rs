trait SomeTrait {
    fn foo(&self);
}
struct S;
impl SomeTrait for S {
    fn foo(&self) {}
}
fn main() {
    let trait_obj: &dyn SomeTrait = &S;

    let &invalid = trait_obj;
    //~^ ERROR E0033
}
