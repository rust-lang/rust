trait FromStructReader<'a> { }
trait ResponseHook {
     fn get(&self);
}
fn foo(res : Box<dyn ResponseHook>) { res.get } //~ ERROR attempted to take value of method
fn main() {}
