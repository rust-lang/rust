// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

fn changer<'a>(mut things: Box<dyn Iterator<Item=&'a mut u8>>) {
    for item in *things { *item = 0 }
    //[current]~^ ERROR the size for values of type
    //[next]~^^ ERROR the trait bound `dyn Iterator<Item = &'a mut u8>: IntoIterator` is not satisfied
    //[next]~| ERROR the size for values of type `<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter` cannot be known at compilation time
    //[next]~| ERROR the type `<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter` is not well-formed
    //[next]~| ERROR `<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter` is not an iterator
    //[next]~| ERROR the type `&mut <dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter` is not well-formed
    //[next]~| ERROR the size for values of type `<<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter as Iterator>::Item` cannot be known at compilation time
    //[next]~| ERROR the type `Option<<<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter as Iterator>::Item>` is not well-formed
    //[next]~| ERROR the size for values of type `<<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter as Iterator>::Item` cannot be known at compilation time
    //[next]~| ERROR type `<<dyn Iterator<Item = &'a mut u8> as IntoIterator>::IntoIter as Iterator>::Item` cannot be dereferenced
    // FIXME(-Ztrait-solver=next): these error messages are horrible and have to be
    // improved before we stabilize the new solver.
}

fn main() {}
