trait FromStructReader<'a> { }
trait ResponseHook {
     fn get(&self);
}
fn foo(res : Box<ResponseHook>) { res.get } //~ ERROR attempted to take value of method
fn main() {}
