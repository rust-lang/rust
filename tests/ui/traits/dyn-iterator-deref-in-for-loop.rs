//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn changer<'a>(mut things: Box<dyn Iterator<Item=&'a mut u8>>) {
    for item in *things { *item = 0 }
    //~^ ERROR `dyn Iterator<Item = &'a mut u8>` is not an iterator
}

fn main() {}
