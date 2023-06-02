// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

fn changer<'a>(mut things: Box<dyn Iterator<Item=&'a mut u8>>) {
    for item in *things { *item = 0 }
    //~^ ERROR the size for values of type
    //[next]~^^ ERROR the type `<_ as IntoIterator>::IntoIter` is not well-formed
    //[next]~| ERROR the trait bound `dyn Iterator<Item = &'a mut u8>: IntoIterator` is not satisfied
}

fn main() {}
